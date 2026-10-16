"""
多GPU并行PDF提取池 (批量优化版)
使用 --pdf_dir 批量处理，避免重复加载模型（30s/次 → 30s/批）
"""
import os
import gc
import glob
import shutil
import tempfile
import subprocess
from typing import Dict, Any, List, Optional, Callable
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from loguru import logger

PDF_EXTRACT_KIT_PATH = "/home/dev_qw/pdf-extract-kit"
CONDA_PYTHON = "/home/dev_qw/miniconda3/envs/pdf-extract-kit-1.0/bin/python"
EXTRACT_SCRIPT = os.path.join(PDF_EXTRACT_KIT_PATH, "extract_pdf.py")


class GPUExtractionPool:
    """
    多GPU并行PDF提取池

    核心优化: 将论文分组为批次，每批用 --pdf_dir 一次性处理，
    模型只加载一次（30s），而不是每篇论文都加载。
    """

    def __init__(self, num_gpus: int = 8, timeout_per_paper: int = 300,
                 batch_size: int = 50):
        """
        Args:
            num_gpus: GPU数量
            timeout_per_paper: 每篇论文超时秒数，用于计算批次总超时
            batch_size: 每批论文数量（模型加载一次处理多少篇）
        """
        self.num_gpus = num_gpus
        self.timeout_per_paper = timeout_per_paper
        self.batch_size = batch_size

    def extract_all(
        self,
        tasks: List[Dict[str, str]],
        progress_callback: Optional[Callable] = None
    ) -> List[Dict[str, Any]]:
        """
        在多张GPU上并行批量提取PDF

        流程:
        1. 将 tasks 按 round-robin 分配给 num_gpus 个 GPU
        2. 每个 GPU 上的 tasks 进一步按 batch_size 分批
        3. 每批: 创建临时目录(symlink) → --pdf_dir 子进程处理 → 整理输出
        4. 8 个 GPU 线程并行执行各自的任务队列

        Args:
            tasks: [{"pdf_path": str, "output_dir": str, "paper_name": str}, ...]
            progress_callback: 每完成一篇时的回调

        Returns:
            所有论文的提取结果列表
        """
        # 跳过已提取的论文（输出目录中已有.md文件）
        pending_tasks = []
        skipped_results = []
        for task in tasks:
            safe_name = task['paper_name'].replace('/', '_')
            paper_output_dir = os.path.join(task["output_dir"], task["paper_name"])
            # 检查 .md 或 .npy（用 safe_name 因为输出文件名基于 sanitized name）
            expected_md = os.path.join(paper_output_dir, f"{safe_name}.md")
            expected_npy = os.path.join(paper_output_dir, f"{safe_name}.npy")
            if os.path.exists(expected_md) or os.path.exists(expected_npy):
                skipped_results.append(self._collect_output(task, -1))
            else:
                pending_tasks.append(task)

        skipped = len(skipped_results)
        total = len(pending_tasks)
        logger.info(
            f"[GPU Pool] 跳过 {skipped} 篇已提取, 待处理 {total} 篇, "
            f"{self.num_gpus} GPUs, 批大小={self.batch_size}"
        )

        # 按 round-robin 分配给各 GPU (仅分配待处理的任务)
        gpu_queues: Dict[int, List[Dict]] = {i: [] for i in range(self.num_gpus)}
        for i, task in enumerate(pending_tasks):
            gpu_id = i % self.num_gpus
            gpu_queues[gpu_id].append(task)

        for gid, queue in gpu_queues.items():
            logger.info(f"  GPU {gid}: {len(queue)} 篇论文")

        all_results: List[Dict[str, Any]] = []
        completed = [0]  # 用列表以便在闭包中修改
        lock_cb = __import__('threading').Lock()

        def on_paper_done(result):
            if progress_callback:
                progress_callback(result)

        with ThreadPoolExecutor(max_workers=self.num_gpus) as executor:
            futures = {}
            for gpu_id, queue in gpu_queues.items():
                if queue:
                    future = executor.submit(
                        self._process_gpu_queue, gpu_id, queue, on_paper_done
                    )
                    futures[future] = gpu_id

            for future in as_completed(futures):
                gpu_id = futures[future]
                try:
                    results = future.result()
                    all_results.extend(results)
                except Exception as e:
                    logger.error(f"[GPU {gpu_id}] 队列执行异常: {e}")

        all_results = skipped_results + all_results
        success = sum(1 for r in all_results if r.get("status") == "success")
        logger.info(f"[GPU Pool] 全部完成: {success}/{total + skipped} 成功 (含跳过 {skipped})")
        return all_results

    def _process_gpu_queue(
        self,
        gpu_id: int,
        queue: List[Dict],
        paper_callback: Callable
    ) -> List[Dict]:
        """处理分配给某个GPU的所有任务（分批执行）"""
        results = []

        # 分批
        batches = [
            queue[i:i + self.batch_size]
            for i in range(0, len(queue), self.batch_size)
        ]

        logger.info(f"[GPU {gpu_id}] 开始处理 {len(queue)} 篇, 分 {len(batches)} 批")

        for batch_idx, batch in enumerate(batches):
            logger.info(
                f"[GPU {gpu_id}] 批次 {batch_idx+1}/{len(batches)}, "
                f"{len(batch)} 篇论文"
            )
            batch_results = self._process_batch(gpu_id, batch)
            results.extend(batch_results)

            for r in batch_results:
                paper_callback(r)

        return results

    def _process_batch(
        self,
        gpu_id: int,
        batch: List[Dict]
    ) -> List[Dict]:
        """使用 --pdf_dir 处理一批论文（一次模型加载）"""
        device = f"cuda:{gpu_id}"
        output_dir = batch[0]["output_dir"]
        abs_output = os.path.abspath(output_dir)

        # 创建临时目录，用 symlink 链接所有 PDF
        temp_dir = tempfile.mkdtemp(prefix=f"gpu{gpu_id}_batch_")
        valid_tasks = []

        for task in batch:
            pdf_path = task["pdf_path"]
            if not os.path.exists(pdf_path):
                continue
            # symlink 到临时目录，文件名用 sanitized paper_name
            safe_name = task['paper_name'].replace('/', '_')
            link_name = os.path.join(temp_dir, f"{safe_name}.pdf")
            try:
                os.makedirs(os.path.dirname(link_name), exist_ok=True)
                os.symlink(os.path.abspath(pdf_path), link_name)
                task["safe_name"] = safe_name
                valid_tasks.append(task)
            except Exception as e:
                logger.warning(f"无法创建symlink {link_name}: {e}")

        if not valid_tasks:
            shutil.rmtree(temp_dir, ignore_errors=True)
            return []

        # 计算超时: 模型加载60s + 每篇处理时间
        timeout = 120 + len(valid_tasks) * self.timeout_per_paper

        cmd = [
            CONDA_PYTHON, EXTRACT_SCRIPT,
            "--pdf_dir", temp_dir,
            "--output", abs_output,
            "--device", "cuda:0"  # CUDA_VISIBLE_DEVICES 限制只看到一个GPU，始终用 cuda:0
        ]

        logger.debug(f"[GPU {gpu_id}] 执行: --pdf_dir {temp_dir} ({len(valid_tasks)} PDFs)")
        # 设置 CUDA_VISIBLE_DEVICES 让进程只看到分配的 GPU，
        # 解决 PaddleOCR 等组件忽略 --device 参数的问题
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

        try:
            proc_result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=PDF_EXTRACT_KIT_PATH,
                env=env
            )

            if proc_result.returncode != 0:
                logger.warning(
                    f"[GPU {gpu_id}] 批次子进程返回非零退出码 {proc_result.returncode}, "
                    f"stderr: {proc_result.stderr[:2000]}"
                )

        except subprocess.TimeoutExpired:
            logger.error(f"[GPU {gpu_id}] 批次超时 ({timeout}s)")
        finally:
            # 清理临时目录
            shutil.rmtree(temp_dir, ignore_errors=True)

        # 批处理输出是扁平结构（output/xxx.npy, output/xxx_images/），
        # 需要重组为子目录结构（output/xxx/xxx.npy, output/xxx/images/）
        self._reorganize_batch_output(abs_output, valid_tasks)

        # 检查每篇论文的输出是否存在，决定成功还是需要回退
        results = []
        failed_tasks = []
        for task in valid_tasks:
            result = self._collect_output(task, gpu_id)
            if result.get("status") == "success":
                results.append(result)
            else:
                failed_tasks.append(task)

        if failed_tasks:
            logger.info(
                f"[GPU {gpu_id}] 批次中 {len(failed_tasks)}/{len(valid_tasks)} 篇无输出, 回退逐篇处理"
            )
            fallback_results = self._fallback_sequential(gpu_id, failed_tasks)
            results.extend(fallback_results)

        return results

    def _reorganize_batch_output(self, output_dir: str, tasks: List[Dict]):
        """将批处理的扁平输出重组为子目录结构"""
        for task in tasks:
            paper_name = task["paper_name"]
            safe_name = task.get("safe_name", paper_name.replace('/', '_'))
            paper_subdir = os.path.join(output_dir, paper_name)

            # 扁平文件路径（pdf-extract-kit 用 symlink 文件名，即 safe_name）
            flat_npy = os.path.join(output_dir, f"{safe_name}.npy")
            flat_images = os.path.join(output_dir, f"{safe_name}_images")

            # 如果扁平文件不存在，跳过
            if not os.path.exists(flat_npy) and not os.path.exists(flat_images):
                continue

            # 创建子目录
            os.makedirs(paper_subdir, exist_ok=True)

            # 移动 .npy 文件（重命名为 paper_name.npy）
            target_npy = os.path.join(paper_subdir, f"{paper_name.replace('/', '_')}.npy")
            if os.path.exists(flat_npy) and not os.path.exists(target_npy):
                shutil.move(flat_npy, target_npy)

            # 移动图片文件夹内容到 images/
            target_images = os.path.join(paper_subdir, "images")
            if os.path.exists(flat_images) and os.path.isdir(flat_images):
                if not os.path.exists(target_images):
                    shutil.move(flat_images, target_images)
                else:
                    # 合并
                    for f in os.listdir(flat_images):
                        src = os.path.join(flat_images, f)
                        dst = os.path.join(target_images, f)
                        if not os.path.exists(dst):
                            shutil.move(src, dst)
                    shutil.rmtree(flat_images, ignore_errors=True)

    def _fallback_sequential(
        self,
        gpu_id: int,
        tasks: List[Dict]
    ) -> List[Dict]:
        """批量失败时回退为逐篇处理"""
        logger.info(f"[GPU {gpu_id}] 回退逐篇处理 {len(tasks)} 篇")
        results = []

        for task in tasks:
            pdf_path = task["pdf_path"]
            paper_name = task["paper_name"]
            safe_name = paper_name.replace('/', '_')
            output_dir = task["output_dir"]

            paper_output_dir = os.path.join(output_dir, paper_name)
            os.makedirs(paper_output_dir, exist_ok=True)

            # 用 symlink 到临时文件确保文件名无斜杠
            safe_pdf = os.path.join(paper_output_dir, f"{safe_name}.pdf")
            if not os.path.exists(safe_pdf) and os.path.exists(pdf_path):
                os.symlink(os.path.abspath(pdf_path), safe_pdf)

            timeout = max(300, self.timeout_per_paper)
            cmd = [
                CONDA_PYTHON, EXTRACT_SCRIPT,
                "--pdf", os.path.abspath(safe_pdf) if os.path.exists(safe_pdf) else os.path.abspath(pdf_path),
                "--output", os.path.abspath(paper_output_dir),
                "--device", "cuda:0"  # CUDA_VISIBLE_DEVICES 限制后始终用 cuda:0
            ]

            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

            try:
                proc_result = subprocess.run(
                    cmd, capture_output=True, text=True,
                    timeout=timeout, cwd=PDF_EXTRACT_KIT_PATH,
                    env=env
                )
                if proc_result.returncode != 0:
                    logger.warning(
                        f"[GPU {gpu_id}] 逐篇处理 {paper_name} 返回码 {proc_result.returncode}, "
                        f"stderr: {proc_result.stderr[:200]}"
                    )
            except subprocess.TimeoutExpired:
                results.append({
                    "status": "error",
                    "message": f"超时(GPU{gpu_id})",
                    "paper_name": paper_name,
                    "pdf_path": pdf_path
                })
                continue
            except Exception as e:
                results.append({
                    "status": "error",
                    "message": str(e),
                    "paper_name": paper_name,
                    "pdf_path": pdf_path
                })
                continue

            # 通过输出文件判断成功与否，而非依赖returncode
            result = self._collect_output(task, gpu_id)
            results.append(result)

        return results

    def _collect_output(self, task: Dict, gpu_id: int) -> Dict:
        """收集单篇论文的提取结果"""
        paper_name = task["paper_name"]
        safe_name = paper_name.replace('/', '_')
        output_dir = task["output_dir"]
        paper_output_dir = os.path.join(output_dir, paper_name)
        images_folder = os.path.join(paper_output_dir, "images")

        # 查找输出文件（用 safe_name 匹配，因为 pdf-extract-kit 基于 symlink 文件名输出）
        expected_npy = os.path.join(paper_output_dir, f"{safe_name}.npy")
        expected_md = os.path.join(paper_output_dir, f"{safe_name}.md")
        expected_images = os.path.join(paper_output_dir, f"{safe_name}_images")

        if not os.path.exists(expected_npy):
            npy_files = list(Path(paper_output_dir).glob("*.npy"))
            if npy_files:
                expected_npy = str(npy_files[0])

        if not os.path.exists(expected_md):
            md_files = list(Path(paper_output_dir).glob("*.md"))
            if md_files:
                expected_md = str(md_files[0])

        if not os.path.exists(expected_images):
            for item in os.listdir(paper_output_dir):
                item_path = os.path.join(paper_output_dir, item)
                if os.path.isdir(item_path) and "image" in item.lower():
                    expected_images = item_path
                    break

        # 重命名图片文件夹
        os.makedirs(images_folder, exist_ok=True)
        if os.path.exists(expected_images) and expected_images != images_folder:
            # 合并内容
            for f in os.listdir(expected_images):
                src = os.path.join(expected_images, f)
                dst = os.path.join(images_folder, f)
                if not os.path.exists(dst):
                    shutil.move(src, dst)

        # 统计图片
        image_count = 0
        if os.path.exists(images_folder):
            image_count = len([f for f in os.listdir(images_folder) if f.endswith('.png')])

        # 读取Markdown
        markdown_content = ""
        md_exists = os.path.exists(expected_md)
        if md_exists:
            with open(expected_md, 'r', encoding='utf-8') as f:
                markdown_content = f.read()
        elif os.path.exists(expected_npy):
            import numpy as np
            markdown_content = str(np.load(expected_npy, allow_pickle=True))
            md_save_path = os.path.join(paper_output_dir, f"{safe_name}.md")
            with open(md_save_path, 'w', encoding='utf-8') as f:
                f.write(markdown_content)
            expected_md = md_save_path
            md_exists = True

        if not md_exists and not os.path.exists(expected_npy):
            return {
                "status": "error",
                "message": "输出文件未生成",
                "paper_name": paper_name,
                "pdf_path": task["pdf_path"],
                "gpu_id": gpu_id
            }

        return {
            "status": "success",
            "paper_name": paper_name,
            "output_folder": paper_output_dir,
            "md_path": expected_md if md_exists else None,
            "npy_path": expected_npy if os.path.exists(expected_npy) else None,
            "images_folder": images_folder,
            "image_count": image_count,
            "markdown_length": len(markdown_content),
            "markdown_content": markdown_content[:1000] + "..." if len(markdown_content) > 1000 else markdown_content,
            "gpu_id": gpu_id
        }

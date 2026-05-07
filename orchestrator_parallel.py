"""
并行协调器 - 使用多GPU和多进程加速数据处理流水线（含阶段重叠）
"""
from typing import Dict, Any, Optional, List
from loguru import logger
import os
import threading
from datetime import datetime

from orchestrator import DataCrawlerOrchestrator


class ParallelDataCrawlerOrchestrator(DataCrawlerOrchestrator):
    """
    并行数据爬取协调器

    与父类相同的Agent接口，但使用:
    - 多GPU并行提取PDF
    - multiprocessing并行清洗和引用图构建
    - 阶段重叠: GPU提取的同时并行清洗已完成的论文
    """

    def __init__(
        self,
        context_manager: Optional[Any] = None,
        num_gpus: int = 8,
        num_cpu_workers: int = 32
    ):
        self.num_gpus = num_gpus
        self.num_cpu_workers = num_cpu_workers
        super().__init__(context_manager)
        logger.info(
            f"ParallelDataCrawlerOrchestrator 初始化: "
            f"{num_gpus} GPUs, {num_cpu_workers} CPU workers"
        )

    def execute_batch_pipeline(
        self,
        task_config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        并行批量下载流水线（含阶段重叠）

        流程:
        1. ArXiv筛选 → 筛选指定分类的论文 (串行，快)
        2. 论文收集 → 下载PDF + 多GPU并行提取 → 生成节点
        3. 数据清洗 → 多进程并行清洗（与GPU提取重叠）
        4. 引用图构建 → 多进程并行提取引用、匹配 → 生成边
        5. 数据集封装 → 整合节点和边

        Args:
            task_config: 任务配置

        Returns:
            Dict[str, Any]: 完整的执行结果
        """
        stage_overlap = task_config.get("stage_overlap", True)

        logger.info("=" * 50)
        logger.info("开始执行并行批量流水线")
        logger.info(f"GPU数: {self.num_gpus}, CPU workers: {self.num_cpu_workers}")
        logger.info(f"阶段重叠: {'开启' if stage_overlap else '关闭'}")
        logger.info("=" * 50)

        self.current_status = "running"
        start_time = datetime.now()

        pipeline_result = {
            "status": "success",
            "start_time": start_time.isoformat(),
            "end_time": None,
            "duration_seconds": 0,
            "stages": {},
            "final_output": {},
            "errors": []
        }

        try:
            # ==================== 阶段0: ArXiv批量筛选 ====================
            logger.info("\n[阶段 0/4] ArXiv批量下载Agent开始执行...")
            stage0_task = {
                "metadata_path": task_config.get(
                    "metadata_path",
                    "/home/dev_qw/DataCrawlerAgent/arxiv-metadata-cleaned.json"
                ),
                "category": task_config.get("category", "cs.AI"),
                "limit": task_config.get("limit", 20),
                "output_dir": task_config.get("output_dirs", {}).get("papers", "./data/papers")
            }

            stage0_result = self.agents['arxiv_batch_downloader'].execute(stage0_task)
            pipeline_result["stages"]["arxiv_batch_download"] = stage0_result

            if stage0_result.get("status") != "success":
                raise Exception(f"ArXiv批量筛选阶段失败: {stage0_result.get('errors', [])}")

            logger.info(f"[阶段 0/4] 完成 - 筛选到 {stage0_result['total_papers_found']} 篇论文")

            paper_sources = stage0_result.get("paper_sources", [])
            if not paper_sources:
                raise Exception("没有找到符合条件的论文")

            # ==================== 阶段1: 并行下载 + 多GPU提取 ====================
            logger.info(f"\n[阶段 1/4] 论文收集 (多GPU并行提取)...")
            stage1_task = {
                "paper_sources": paper_sources,
                "output_dir": task_config.get("output_dirs", {}).get("papers", "./data/papers"),
                "extract_multimodal": task_config.get("extract_multimodal", True),
                "device": f"cuda:0",
                "parallel_extraction": True,
                "num_gpus": self.num_gpus,
                "batch_size": task_config.get("parallel", {}).get("batch_size", 20),
                "batch_download": True
            }

            if stage_overlap:
                # 阶段重叠模式: 在GPU提取的回调中启动清洗
                stage1_result = self._execute_with_overlap(stage1_task, task_config)
            else:
                stage1_result = self.agents['paper_collector'].execute(stage1_task)

            pipeline_result["stages"]["paper_collection"] = stage1_result

            if stage1_result.get("status") != "success":
                raise Exception(f"论文收集阶段失败: {stage1_result.get('errors', [])}")

            logger.info(
                f"[阶段 1/4] 完成 - 收集了 {len(stage1_result.get('papers', []))} 篇论文"
            )

            if not stage_overlap:
                # ==================== 阶段2: 并行数据清洗 ====================
                logger.info(f"\n[阶段 2/4] 并行数据清洗 ({self.num_cpu_workers} workers)...")
                stage2_task = {
                    "parallel": True,
                    "num_workers": self.num_cpu_workers,
                    "output_dir": task_config.get("output_dirs", {}).get("cleaned", "./data/cleaned")
                }

                stage2_result = self.agents['data_cleaner'].execute(
                    task=stage2_task,
                    upstream_data=stage1_result
                )
                pipeline_result["stages"]["data_cleaning"] = stage2_result

                logger.info(
                    f"[阶段 2/4] 完成 - 清洗了 {stage2_result.get('stats', {}).get('total_cleaned', 0)} 篇论文"
                )

                # ==================== 阶段3: 并行引用图构建 ====================
                logger.info(f"\n[阶段 3/4] 并行引用图构建 ({self.num_cpu_workers} workers)...")
                stage3_task = {
                    "metadata_path": task_config.get(
                        "metadata_path",
                        "/home/dev_qw/DataCrawlerAgent/arxiv-metadata-cleaned.json"
                    ),
                    "output_dir": task_config.get("output_dirs", {}).get("graphs", "./data/graphs"),
                    "parallel": True,
                    "num_workers": self.num_cpu_workers
                }

                stage3_result = self.agents['reference_graph_builder'].execute(
                    task=stage3_task,
                    upstream_data=stage1_result
                )
                pipeline_result["stages"]["reference_graph_building"] = stage3_result

                if stage3_result.get("status") != "success":
                    logger.warning(f"引用图构建阶段部分失败: {stage3_result.get('message', '未知错误')}")
                else:
                    logger.info(f"[阶段 3/4] 完成 - 构建了 {stage3_result.get('edge_count', 0)} 条引用边")
            else:
                # 阶段重叠模式下，清洗和引用图已经在提取过程中完成
                stage2_result = self._overlap_clean_result or {}
                stage3_result = self._overlap_ref_result or {}
                pipeline_result["stages"]["data_cleaning"] = stage2_result
                pipeline_result["stages"]["reference_graph_building"] = stage3_result
                logger.info(f"[阶段 2-3/4] 已在提取过程中重叠完成")

            # ==================== 阶段4: 数据集封装 ====================
            logger.info("\n[阶段 4/4] 数据集封装Agent开始执行...")
            stage4_task = {
                "dataset_name": task_config.get(
                    "dataset_name",
                    f"multimodal_graph_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                ),
                "output_format": task_config.get("output_format", "json"),
                "output_dir": task_config.get("output_dirs", {}).get("datasets", "./data/datasets"),
                "include_metadata": task_config.get("include_metadata", True),
                "edges_file": (
                    stage3_result.get("edges_file", "")
                    if stage3_result.get("status") == "success" else None
                )
            }

            stage4_result = self.agents['dataset_formatter'].execute(
                task=stage4_task,
                upstream_data=stage1_result
            )
            pipeline_result["stages"]["dataset_formatting"] = stage4_result

            if stage4_result.get("status") != "success":
                raise Exception(f"数据集封装阶段失败: {stage4_result.get('message', '未知错误')}")

            logger.info(f"[阶段 4/4] 完成 - 数据集已保存到: {stage4_result.get('output_path', '')}")

            # 记录最终输出
            pipeline_result["final_output"] = {
                "dataset_path": stage4_result.get("output_path", ""),
                "nodes_path": stage4_result.get("nodes_path", ""),
                "edges_path": (
                    stage3_result.get("edges_file", "")
                    if stage3_result.get("status") == "success" else ""
                ),
                "node_count": len(stage1_result.get('papers', [])),
                "edge_count": (
                    stage3_result.get("edge_count", 0)
                    if stage3_result.get("status") == "success" else 0
                ),
                "modality_distribution": stage4_result.get("stats", {}).get("modality_distribution", {})
            }

        except Exception as e:
            pipeline_result["status"] = "failed"
            pipeline_result["errors"].append(str(e))
            logger.error(f"流水线执行失败: {e}")

        finally:
            end_time = datetime.now()
            pipeline_result["end_time"] = end_time.isoformat()
            pipeline_result["duration_seconds"] = (end_time - start_time).total_seconds()
            self.current_status = "completed"
            self.execution_history.append(pipeline_result)

        logger.info("\n" + "=" * 50)
        logger.info(f"并行流水线执行完成 - 状态: {pipeline_result['status']}")
        logger.info(f"节点数: {pipeline_result['final_output'].get('node_count', 0)}")
        logger.info(f"边数: {pipeline_result['final_output'].get('edge_count', 0)}")
        logger.info(f"总耗时: {pipeline_result['duration_seconds']:.2f} 秒")
        logger.info("=" * 50)

        return pipeline_result

    def _execute_with_overlap(
        self,
        stage1_task: Dict[str, Any],
        task_config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        阶段重叠模式: GPU提取的同时，在后台线程中清洗已完成提取的论文

        流程:
        1. 启动GPU提取（主线程）
        2. 每完成一篇论文 → 加入清洗队列
        3. 后台线程从队列中取论文进行清洗
        4. 全部提取完成后 → 对已清洗数据构建引用图
        """
        import queue
        from concurrent.futures import ThreadPoolExecutor

        cleaned_papers = []
        clean_errors = []
        clean_queue = queue.Queue()

        def clean_worker():
            """后台清洗 worker"""
            from agents.data_cleaner import DataCleanerAgent
            agent = DataCleanerAgent(
                client=self.client,
                system_prompt=self.data_cleaner_prompt,
                mcp_tool_manager=self.mcp_tool_manager
            )
            while True:
                item = clean_queue.get()
                if item is None:  # sentinel
                    clean_queue.task_done()
                    break
                try:
                    result = agent._clean_paper_directory(item)
                    if result:
                        cleaned_papers.append(result)
                except Exception as e:
                    clean_errors.append(str(e))
                clean_queue.task_done()

        # 启动清洗后台线程
        clean_thread = threading.Thread(target=clean_worker, daemon=True)
        clean_thread.start()

        def on_paper_extracted(result):
            """每篇论文提取完成后的回调 → 送入清洗队列"""
            if result.get("status") == "success" and result.get("output_folder"):
                clean_queue.put(result["output_folder"])

        # 注入进度回调到 stage1 task
        stage1_task["progress_callback"] = on_paper_extracted

        # 执行GPU提取（主线程，阻塞直到完成）
        stage1_result = self.agents['paper_collector'].execute(stage1_task)

        # 发送终止信号
        clean_queue.put(None)
        clean_thread.join(timeout=300)

        logger.info(f"[阶段重叠] 清洗完成: {len(cleaned_papers)} 篇, 错误: {len(clean_errors)}")

        # 保存清洗结果供后续使用
        self._overlap_clean_result = {
            "status": "success",
            "stats": {
                "total_cleaned": len(cleaned_papers),
                "errors": len(clean_errors)
            },
            "cleaned_papers": cleaned_papers
        }

        # 引用图构建（在全部提取完成后执行）
        logger.info(f"\n[阶段重叠] 开始引用图构建...")
        stage3_task = {
            "metadata_path": task_config.get(
                "metadata_path",
                "/home/dev_qw/DataCrawlerAgent/arxiv-metadata-cleaned.json"
            ),
            "output_dir": task_config.get("output_dirs", {}).get("graphs", "./data/graphs"),
            "parallel": True,
            "num_workers": self.num_cpu_workers
        }

        stage3_result = self.agents['reference_graph_builder'].execute(
            task=stage3_task,
            upstream_data=stage1_result
        )
        self._overlap_ref_result = stage3_result

        if stage3_result.get("status") != "success":
            logger.warning(f"引用图构建阶段部分失败: {stage3_result.get('message', '未知错误')}")
        else:
            logger.info(f"[阶段重叠] 引用图构建完成 - {stage3_result.get('edge_count', 0)} 条边")

        return stage1_result


def create_parallel_orchestrator(num_gpus=8, num_cpu_workers=8) -> ParallelDataCrawlerOrchestrator:
    return ParallelDataCrawlerOrchestrator(num_gpus=num_gpus, num_cpu_workers=num_cpu_workers)

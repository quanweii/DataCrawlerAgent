"""
PDF多模态数据提取工具 - MCP工具封装
使用subprocess调用 pdf-extract-kit 进行论文PDF的多模态数据提取

输出结构:
    output_dir/
    └── {paper_name}/              # 以论文名称命名的文件夹
        ├── {paper_name}.md        # Markdown文件
        ├── {paper_name}.npy       # NPY格式数据
        └── images/                # 图片文件夹
            ├── 0.png
            ├── 1.png
            └── ...
"""
import os
import sys
import gc
import subprocess
import shutil
from typing import Dict, Any, Optional, List
from pathlib import Path
from loguru import logger

# pdf-extract-kit 路径
PDF_EXTRACT_KIT_PATH = os.environ.get("PDF_EXTRACT_KIT_PATH", "")


class PDFExtractorTool:
    """PDF多模态数据提取工具（使用subprocess调用）"""

    def __init__(self, config: Optional[Dict] = None):
        """
        初始化PDF提取工具

        Args:
            config: 配置字典，可选
        """
        self.config = config or {}

    def extract(
        self,
        pdf_path: str,
        output_dir: str,
        paper_name: Optional[str] = None,
        save_markdown: bool = True,
        save_npy: bool = True,
        device: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        从PDF提取多模态数据（使用subprocess调用pdf-extract-kit）

        Args:
            pdf_path: PDF文件路径
            output_dir: 输出根目录
            paper_name: 论文名称（用于命名输出文件夹），默认使用PDF文件名
            save_markdown: 是否保存Markdown文件
            save_npy: 是否保存NPY文件
            device: 计算设备，如 'cuda:1' 或 'cpu'

        Returns:
            Dict: 提取结果
        """
        if not PDF_EXTRACT_KIT_PATH or not (Path(PDF_EXTRACT_KIT_PATH) / "extract_pdf.py").is_file():
            return {"status": "error", "message": "Configure PDF_EXTRACT_KIT_PATH and optional PDF_EXTRACT_PYTHON for the external arXiv extraction backend."}

        # 验证输入文件
        if not os.path.exists(pdf_path):
            return {
                "status": "error",
                "message": f"PDF文件不存在: {pdf_path}"
            }

        # 获取论文名称
        if paper_name is None:
            paper_name = Path(pdf_path).stem

        # 创建以论文名称命名的输出文件夹
        paper_output_dir = os.path.join(output_dir, paper_name)
        os.makedirs(paper_output_dir, exist_ok=True)

        # 图片文件夹
        images_folder = os.path.join(paper_output_dir, "images")
        os.makedirs(images_folder, exist_ok=True)

        device = device or self.config.get("device", "cuda:1")

        logger.info(f"开始提取PDF: {pdf_path}")
        logger.info(f"输出目录: {paper_output_dir}")

        try:
            # 使用subprocess调用pdf-extract-kit
            # 转换为绝对路径，因为subprocess的cwd是pdf-extract-kit目录
            abs_pdf_path = os.path.abspath(pdf_path)
            abs_output_dir = os.path.abspath(paper_output_dir)

            conda_python = os.environ.get("PDF_EXTRACT_PYTHON", sys.executable)
            extract_script = os.path.join(PDF_EXTRACT_KIT_PATH, "extract_pdf.py")
            cmd = [
                conda_python, extract_script,
                "--pdf", abs_pdf_path,
                "--output", abs_output_dir,
                "--device", device
            ]

            logger.info(f"执行命令: {' '.join(cmd)}")

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=1800,  # 30分钟超时
                cwd=PDF_EXTRACT_KIT_PATH  # 在pdf-extract-kit目录下执行
            )

            if result.returncode != 0:
                logger.error(f"PDF提取失败: {result.stderr}")
                return {
                    "status": "error",
                    "message": f"PDF提取失败: {result.stderr}",
                    "paper_name": paper_name,
                    "pdf_path": pdf_path
                }

            logger.info(f"PDF提取完成: {result.stdout}")

            # 查找并整理输出文件
            # pdf-extract-kit 输出: {paper_name}.npy, {paper_name}.md, {paper_name}_images/
            expected_npy = os.path.join(paper_output_dir, f"{paper_name}.npy")
            expected_md = os.path.join(paper_output_dir, f"{paper_name}.md")
            expected_images = os.path.join(paper_output_dir, f"{paper_name}_images")

            # 如果期望的文件不存在，尝试查找
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

            # 重命名图片文件夹为标准格式 "images"
            if os.path.exists(expected_images) and expected_images != images_folder:
                if os.path.exists(images_folder):
                    shutil.rmtree(images_folder)
                shutil.move(expected_images, images_folder)
                expected_images = images_folder

            # 统计图片数量
            image_count = 0
            if os.path.exists(images_folder):
                image_count = len([f for f in os.listdir(images_folder) if f.endswith('.png')])

            # 读取Markdown内容（优先从MD文件，否则从NPY文件）
            markdown_content = ""
            if os.path.exists(expected_md):
                with open(expected_md, 'r', encoding='utf-8') as f:
                    markdown_content = f.read()
            elif os.path.exists(expected_npy):
                # 从NPY文件读取并保存为MD文件
                import numpy as np
                markdown_content = str(np.load(expected_npy, allow_pickle=True))
                # 保存MD文件
                md_save_path = os.path.join(paper_output_dir, f"{paper_name}.md")
                with open(md_save_path, 'w', encoding='utf-8') as f:
                    f.write(markdown_content)
                expected_md = md_save_path
                logger.info(f"已从NPY生成MD文件: {md_save_path}")

            # 清理内存
            gc.collect()

            result = {
                "status": "success",
                "paper_name": paper_name,
                "output_folder": paper_output_dir,
                "md_path": expected_md if os.path.exists(expected_md) else None,
                "npy_path": expected_npy if os.path.exists(expected_npy) else None,
                "images_folder": images_folder,
                "image_count": image_count,
                "markdown_length": len(markdown_content),
                "markdown_content": markdown_content[:1000] + "..." if len(markdown_content) > 1000 else markdown_content
            }

            logger.info(f"PDF提取完成: {image_count}张图片, {len(markdown_content)}字符")
            return result

        except subprocess.TimeoutExpired:
            logger.error("PDF提取超时")
            return {
                "status": "error",
                "message": "PDF提取超时（超过10分钟）",
                "paper_name": paper_name,
                "pdf_path": pdf_path
            }
        except Exception as e:
            logger.error(f"PDF提取失败: {e}")
            return {
                "status": "error",
                "message": str(e),
                "paper_name": paper_name,
                "pdf_path": pdf_path
            }

    def batch_extract(
        self,
        pdf_paths: List[str],
        output_dir: str,
        device: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        批量提取多个PDF文件

        Args:
            pdf_paths: PDF文件路径列表
            output_dir: 输出根目录
            device: 计算设备

        Returns:
            List[Dict]: 所有PDF的提取结果列表
        """
        results = []
        for pdf_path in pdf_paths:
            result = self.extract(
                pdf_path=pdf_path,
                output_dir=output_dir,
                device=device
            )
            results.append(result)
        return results


# ==================== MCP工具函数 ====================

def extract_pdf_to_multimodal(
    pdf_path: str,
    output_dir: str = "./data/extracted",
    paper_name: Optional[str] = None,
    device: str = "cuda:1"
) -> Dict[str, Any]:
    """
    MCP工具函数: 从PDF提取多模态数据

    Args:
        pdf_path: PDF文件路径
        output_dir: 输出根目录
        paper_name: 论文名称（可选）
        device: 计算设备

    Returns:
        Dict: 提取结果
    """
    tool = PDFExtractorTool()
    return tool.extract(
        pdf_path=pdf_path,
        output_dir=output_dir,
        paper_name=paper_name,
        device=device
    )


def batch_extract_pdfs(
    pdf_paths: List[str],
    output_dir: str = "./data/extracted",
    device: str = "cuda:1"
) -> List[Dict[str, Any]]:
    """
    MCP工具函数: 批量提取多个PDF

    Args:
        pdf_paths: PDF文件路径列表
        output_dir: 输出根目录
        device: 计算设备

    Returns:
        List[Dict]: 提取结果列表
    """
    tool = PDFExtractorTool()
    return tool.batch_extract(
        pdf_paths=pdf_paths,
        output_dir=output_dir,
        device=device
    )


# ==================== 测试代码 ====================
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="PDF多模态数据提取工具")
    parser.add_argument("--pdf", type=str, required=True, help="PDF文件路径")
    parser.add_argument("--output", type=str, default="./output", help="输出目录")
    parser.add_argument("--device", type=str, default="cuda:1", help="计算设备")

    args = parser.parse_args()

    result = extract_pdf_to_multimodal(
        pdf_path=args.pdf,
        output_dir=args.output,
        device=args.device
    )

    print("\n" + "=" * 50)
    print("提取结果:")
    print(f"  状态: {result.get('status')}")
    if result.get('status') == 'success':
        print(f"  论文名称: {result.get('paper_name')}")
        print(f"  输出文件夹: {result.get('output_folder')}")
        print(f"  Markdown文件: {result.get('md_path')}")
        print(f"  NPY文件: {result.get('npy_path')}")
        print(f"  图片文件夹: {result.get('images_folder')}")
        print(f"  图片数量: {result.get('image_count')}")
        print(f"  Markdown长度: {result.get('markdown_length')} 字符")
    else:
        print(f"  错误信息: {result.get('message')}")
    print("=" * 50)

#!/usr/bin/env python3
"""
NPY 转 Markdown 转换工具

将 pdf-extract-kit 生成的 .npy 文件转换为 .md 文件
"""
import numpy as np
import os
import sys
from pathlib import Path


def npy_to_md(npy_path: str, output_path: str = None) -> dict:
    """
    将 NPY 文件转换为 Markdown 文件

    Args:
        npy_path: NPY 文件路径
        output_path: 输出 MD 文件路径（可选，默认与 NPY 同目录同名）

    Returns:
        dict: 转换结果
    """
    result = {
        "status": "success",
        "npy_path": npy_path,
        "md_path": None,
        "content_length": 0,
        "error": None
    }

    try:
        # 检查文件是否存在
        if not os.path.exists(npy_path):
            result["status"] = "error"
            result["error"] = f"文件不存在: {npy_path}"
            return result

        # 读取 NPY 文件
        data = np.load(npy_path, allow_pickle=True)

        # 提取 Markdown 内容
        if data.shape == ():
            content = data.item()
        else:
            content = str(data)

        if not isinstance(content, str):
            content = str(content)

        result["content_length"] = len(content)

        # 确定输出路径
        if output_path is None:
            npy_path_obj = Path(npy_path)
            output_path = str(npy_path_obj.with_suffix('.md'))

        # 写入 Markdown 文件
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(content)

        result["md_path"] = output_path
        print(f"✓ 转换成功: {npy_path} -> {output_path} ({len(content)} 字符)")

    except Exception as e:
        result["status"] = "error"
        result["error"] = str(e)
        print(f"✗ 转换失败: {npy_path} - {e}")

    return result


def batch_convert(data_dir: str, pattern: str = "**/*.npy") -> dict:
    """
    批量转换目录下的所有 NPY 文件

    Args:
        data_dir: 数据目录
        pattern: 文件匹配模式

    Returns:
        dict: 批量转换结果统计
    """
    stats = {
        "total": 0,
        "success": 0,
        "failed": 0,
        "skipped": 0,
        "results": []
    }

    data_path = Path(data_dir)
    npy_files = list(data_path.glob(pattern))

    print(f"\n找到 {len(npy_files)} 个 NPY 文件\n")
    print("=" * 60)

    for npy_file in npy_files:
        stats["total"] += 1

        # 检查是否已有 MD 文件
        md_file = npy_file.with_suffix('.md')
        if md_file.exists():
            print(f"○ 跳过（已存在）: {npy_file.name}")
            stats["skipped"] += 1
            continue

        result = npy_to_md(str(npy_file))
        stats["results"].append(result)

        if result["status"] == "success":
            stats["success"] += 1
        else:
            stats["failed"] += 1

    print("\n" + "=" * 60)
    print(f"转换完成: 总计 {stats['total']}, 成功 {stats['success']}, 失败 {stats['failed']}, 跳过 {stats['skipped']}")

    return stats


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="NPY 转 Markdown 工具")
    parser.add_argument("--npy", "-n", type=str, help="单个 NPY 文件路径")
    parser.add_argument("--dir", "-d", type=str, help="批量转换目录")
    parser.add_argument("--output", "-o", type=str, help="输出文件路径（仅单文件模式）")
    parser.add_argument("--force", "-f", action="store_true", help="强制覆盖已存在的 MD 文件")

    args = parser.parse_args()

    if args.npy:
        # 单文件模式
        npy_to_md(args.npy, args.output)
    elif args.dir:
        # 批量模式
        batch_convert(args.dir)
    else:
        # 默认：转换 cs_ai_papers 目录
        default_dir = "/home/QuanWei/DataCrawlerAgent/data/cs_ai_papers"
        if os.path.exists(default_dir):
            batch_convert(default_dir)
        else:
            print("请指定 --npy 或 --dir 参数")
            parser.print_help()

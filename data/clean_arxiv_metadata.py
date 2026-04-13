#!/usr/bin/env python3
"""
arXiv 元数据清洗工具

清洗 arxiv-metadata-oai-snapshot.json 文件：
- 删除标题、摘要等字段中的换行符 \n
- 删除换行符后面多余的空格
- 合并连续多个空格为单个空格
- 去除首尾空白

输入: /home/QuanWei/arxiv-metadata-oai-snapshot.json
输出: /home/QuanWei/arxiv-metadata-cleaned.json
"""
import json
import re
import os
from pathlib import Path
from datetime import datetime


def clean_text(text: str) -> str:
    """
    清洗文本字段

    - 将 \n 替换为空格
    - 合并多个连续空格
    - 去除首尾空格
    """
    if not text or not isinstance(text, str):
        return text

    # 将换行符替换为空格
    text = text.replace('\n', ' ')
    # 合并多个连续空格为单个空格
    text = re.sub(r' +', ' ', text)
    # 去除首尾空格
    text = text.strip()

    return text


def clean_paper(paper: dict) -> dict:
    """
    清洗单篇论文数据
    """
    cleaned = paper.copy()

    # 清洗主要文本字段
    text_fields = ['title', 'abstract', 'comments']

    for field in text_fields:
        if field in cleaned and cleaned[field]:
            cleaned[field] = clean_text(cleaned[field])

    # 清洗作者（可能有换行）
    if 'authors' in cleaned and cleaned['authors']:
        cleaned['authors'] = clean_text(cleaned['authors'])

    return cleaned


def process_file(input_path: str, output_path: str, limit: int = None):
    """
    处理整个文件

    Args:
        input_path: 输入文件路径
        output_path: 输出文件路径
        limit: 限制处理的行数（用于测试）
    """
    total = 0
    cleaned_count = 0
    error_count = 0

    # 统计信息
    title_cleaned = 0
    abstract_cleaned = 0

    print(f"开始处理: {input_path}")
    print(f"输出到: {output_path}")
    print("=" * 60)

    start_time = datetime.now()

    with open(input_path, 'r', encoding='utf-8') as f_in, \
         open(output_path, 'w', encoding='utf-8') as f_out:

        for line_num, line in enumerate(f_in, 1):
            if limit and line_num > limit:
                break

            # 进度显示
            if line_num % 100000 == 0:
                elapsed = (datetime.now() - start_time).total_seconds()
                speed = line_num / elapsed if elapsed > 0 else 0
                print(f"  已处理: {line_num:,} 行, 速度: {speed:.0f} 行/秒")

            line = line.strip()
            if not line:
                continue

            try:
                paper = json.loads(line)

                # 检查是否需要清洗
                original_title = paper.get('title', '')
                original_abstract = paper.get('abstract', '')

                # 清洗
                cleaned_paper = clean_paper(paper)
                total += 1

                # 统计清洗情况
                if original_title != cleaned_paper.get('title', ''):
                    title_cleaned += 1
                if original_abstract != cleaned_paper.get('abstract', ''):
                    abstract_cleaned += 1

                # 写入清洗后的数据
                f_out.write(json.dumps(cleaned_paper, ensure_ascii=False) + '\n')

            except json.JSONDecodeError as e:
                error_count += 1
                if error_count <= 5:
                    print(f"  行 {line_num} JSON 解析错误: {e}")
                continue

    # 输出统计
    elapsed = (datetime.now() - start_time).total_seconds()
    print("\n" + "=" * 60)
    print("处理完成!")
    print(f"  总处理: {total:,} 篇论文")
    print(f"  标题被清洗: {title_cleaned:,} 篇 ({title_cleaned/total*100:.1f}%)")
    print(f"  摘要被清洗: {abstract_cleaned:,} 篇 ({abstract_cleaned/total*100:.1f}%)")
    print(f"  错误: {error_count} 行")
    print(f"  耗时: {elapsed:.1f} 秒")
    print(f"  输出文件: {output_path}")

    # 显示文件大小
    input_size = os.path.getsize(input_path) / (1024**3)
    output_size = os.path.getsize(output_path) / (1024**3)
    print(f"  原始文件: {input_size:.2f} GB")
    print(f"  清洗后: {output_size:.2f} GB")


def main():
    # 默认路径
    input_file = "/home/QuanWei/arxiv-metadata-oai-snapshot.json"
    output_file = "/home/QuanWei/arxiv-metadata-cleaned.json"

    # 检查输入文件
    if not os.path.exists(input_file):
        print(f"错误: 输入文件不存在 - {input_file}")
        return

    # 处理文件
    process_file(input_file, output_file)

    # 验证结果
    print("\n" + "=" * 60)
    print("验证清洗结果...")

    # 读取清洗后的前几条数据
    with open(output_file, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i >= 3:
                break
            paper = json.loads(line)
            print(f"\n论文 {i+1}:")
            print(f"  ID: {paper.get('id')}")
            print(f"  标题: {paper.get('title', '')[:80]}...")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="清洗 arXiv 元数据")
    parser.add_argument("--input", "-i", type=str,
                       default="/home/QuanWei/arxiv-metadata-oai-snapshot.json",
                       help="输入文件路径")
    parser.add_argument("--output", "-o", type=str,
                       default="/home/QuanWei/arxiv-metadata-cleaned.json",
                       help="输出文件路径")
    parser.add_argument("--limit", "-n", type=int, default=None,
                       help="限制处理行数（用于测试）")

    args = parser.parse_args()

    if args.limit:
        process_file(args.input, args.output, args.limit)
    else:
        process_file(args.input, args.output)

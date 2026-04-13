#!/usr/bin/env python3
"""
重新生成引用边 CSV 文件
"""
import os
import sys
import json
import csv

sys.path.insert(0, '/home/QuanWei/DataCrawlerAgent')

from agents.reference_graph_builder import ReferenceGraphBuilderAgent

def get_paper_list(data_dir):
    """获取所有论文目录"""
    papers = []
    for item in os.listdir(data_dir):
        item_path = os.path.join(data_dir, item)
        if os.path.isdir(item_path):
            # 查找 markdown 文件
            md_file = os.path.join(item_path, f"{item}.md")
            if os.path.exists(md_file):
                papers.append({
                    'paper_id': item,
                    'markdown_path': md_file,
                    'title': ''  # 可以从 markdown 中提取
                })
    return papers

def main():
    print("="*60)
    print("重新生成引用边")
    print("="*60)

    # 初始化 agent
    agent = ReferenceGraphBuilderAgent(
        client=None,
        system_prompt=""
    )

    # 加载元数据
    metadata_path = '/home/QuanWei/arxiv-metadata-cs.json'
    print(f"\n加载元数据: {metadata_path}")
    agent.load_arxiv_metadata(metadata_path)

    # 获取论文列表
    data_dir = '/home/QuanWei/DataCrawlerAgent/data/recent_papers'
    papers = get_paper_list(data_dir)
    print(f"\n找到 {len(papers)} 篇论文")

    # 构建引用图
    output_csv = '/home/QuanWei/DataCrawlerAgent/data/graphs/citation_edges.csv'

    result = agent.build_reference_graph(papers, output_csv)

    print("\n" + "="*60)
    print("结果统计:")
    print(f"  处理论文数: {result['stats']['total_papers']}")
    print(f"  有References的论文: {result['stats']['papers_with_references']}")
    print(f"  提取的引用数: {result['stats']['total_references_extracted']}")
    print(f"  匹配的引用数: {result['stats']['matched_references']}")
    print(f"  创建的边数: {result['edge_count']}")
    print(f"\n输出文件: {result['edges_file']}")
    print("="*60)

    # 显示部分结果
    print("\n前 20 条边:")
    with open(output_csv, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            if i >= 20:
                break
            print(f"\n{i+1}. {row['source_id']} -> {row['target_id']}")
            print(f"   源标题: {row['source_title'][:50]}...")
            print(f"   目标标题: {row['target_title'][:50]}...")
            print(f"   引用文本: {row['reference_text'][:60]}...")

if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""
测试引用边提取功能
"""
import os
import sys
import json

# 添加项目路径
sys.path.insert(0, '/home/QuanWei/DataCrawlerAgent')

from agents.reference_graph_builder import ReferenceGraphBuilderAgent

def test_reference_extraction():
    """测试引用提取"""
    # 初始化 agent (不需要实际的 OpenAI client)
    agent = ReferenceGraphBuilderAgent(
        client=None,
        system_prompt=""
    )

    # 加载 arxiv 元数据
    metadata_path = '/home/QuanWei/arxiv-metadata-cs.json'
    print(f"加载元数据: {metadata_path}")
    count = agent.load_arxiv_metadata(metadata_path)
    print(f"已加载 {count} 篇论文的索引")

    # 读取测试论文的 markdown
    paper_path = '/home/QuanWei/DataCrawlerAgent/data/recent_papers/2402.01677/2402.01677.md'
    with open(paper_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # 提取 References 部分
    print("\n" + "="*60)
    print("提取 References 部分...")
    references_text = agent.extract_references_section(content)
    if references_text:
        print(f"References 长度: {len(references_text)} 字符")
        print("\n前 500 字符:")
        print(references_text[:500])
    else:
        print("未找到 References 部分!")
        return

    # 解析参考文献
    print("\n" + "="*60)
    print("解析参考文献...")
    references = agent.parse_references(references_text)
    print(f"解析出 {len(references)} 条参考文献")

    # 显示前 10 条解析结果
    print("\n前 10 条解析结果:")
    for i, ref in enumerate(references[:10]):
        print(f"\n{i+1}. 标题: {ref.get('title', 'N/A')[:80]}...")
        if ref.get('year'):
            print(f"   年份: {ref.get('year')}")

    # 匹配论文
    print("\n" + "="*60)
    print("匹配论文...")
    matched_count = 0
    matched_refs = []

    for ref in references:
        ref_title = ref.get('title', '')
        if not ref_title or len(ref_title) < 10:
            continue

        matched_id = agent.find_matching_paper(ref_title)
        if matched_id:
            matched_count += 1
            matched_refs.append({
                'title': ref_title[:60],
                'matched_id': matched_id,
                'matched_title': agent.id_to_title.get(matched_id, '')[:60]
            })

    print(f"\n成功匹配 {matched_count} 条引用")

    print("\n匹配结果示例 (前 15 条):")
    for i, m in enumerate(matched_refs[:15]):
        print(f"\n{i+1}. 引用标题: {m['title']}...")
        print(f"   匹配ID: {m['matched_id']}")
        print(f"   匹配标题: {m['matched_title']}...")

    return matched_refs

if __name__ == '__main__':
    test_reference_extraction()

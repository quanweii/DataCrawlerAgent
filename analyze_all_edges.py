#!/usr/bin/env python3
"""
分析所有论文的引用边质量
"""
import os
import re
import csv
from collections import defaultdict

def extract_declared_arxiv_ids(md_path):
    """从 markdown 文件中提取明确标注的 arXiv ID"""
    ids = set()
    try:
        with open(md_path, 'r', encoding='utf-8') as f:
            content = f.read()
        # 匹配 arXiv:xxxx.xxxxx 或 arxiv:xxxx.xxxxx 格式
        matches = re.findall(r'arXiv[:\s]+(\d{4}\.\d{4,5})', content, re.IGNORECASE)
        ids = set(matches)
    except Exception as e:
        pass
    return ids

def get_generated_edges(csv_path, source_id):
    """从 CSV 中获取指定论文的生成边"""
    edges = {}
    try:
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row['source_id'] == source_id:
                    edges[row['target_id']] = row['reference_text'][:80]
    except:
        pass
    return edges

def main():
    papers_dir = '/home/QuanWei/DataCrawlerAgent/data/recent_papers'
    csv_path = '/home/QuanWei/DataCrawlerAgent/data/graphs/citation_edges.csv'
    
    # 获取所有论文目录
    paper_ids = []
    for item in os.listdir(papers_dir):
        if os.path.isdir(os.path.join(papers_dir, item)) and not item.startswith('.'):
            md_file = os.path.join(papers_dir, item, f"{item}.md")
            if os.path.exists(md_file):
                paper_ids.append(item)
    
    paper_ids.sort()
    
    print("=" * 80)
    print("引用边质量分析报告")
    print("=" * 80)
    
    total_declared = 0
    total_matched = 0
    total_missed = 0
    total_extra = 0
    
    results = []
    
    for paper_id in paper_ids:
        md_path = os.path.join(papers_dir, paper_id, f"{paper_id}.md")
        
        # 提取明确标注的 arXiv ID
        declared_ids = extract_declared_arxiv_ids(md_path)
        
        # 获取生成的边
        generated_edges = get_generated_edges(csv_path, paper_id)
        generated_ids = set(generated_edges.keys())
        
        # 计算匹配情况
        correctly_matched = declared_ids & generated_ids
        missed = declared_ids - generated_ids
        extra = generated_ids - declared_ids
        
        total_declared += len(declared_ids)
        total_matched += len(correctly_matched)
        total_missed += len(missed)
        total_extra += len(extra)
        
        results.append({
            'paper_id': paper_id,
            'declared': len(declared_ids),
            'generated': len(generated_ids),
            'correct': len(correctly_matched),
            'missed': len(missed),
            'extra': len(extra),
            'missed_ids': missed,
            'extra_ids': extra,
            'extra_edges': {k: v for k, v in generated_edges.items() if k in extra}
        })
    
    # 打印每篇论文的分析结果
    for r in results:
        print(f"\n{'='*80}")
        print(f"论文: {r['paper_id']}")
        print(f"{'='*80}")
        print(f"  明确标注 arXiv 的引用: {r['declared']} 篇")
        print(f"  生成的边数: {r['generated']} 条")
        print(f"  ✅ 正确匹配: {r['correct']} 条")
        print(f"  ❌ 漏连: {r['missed']} 条")
        print(f"  ⚠️ 多余: {r['extra']} 条")
        
        if r['missed'] > 0:
            print(f"\n  漏连的 arXiv ID:")
            for mid in r['missed_ids']:
                print(f"    - {mid}")
        
        if r['extra'] > 0 and r['extra'] <= 10:
            print(f"\n  多余的边 (可能是在 arXiv 上但未标注的引用):")
            for eid, ref_text in r['extra_edges'].items():
                print(f"    - {eid}: {ref_text}...")
    
    # 打印总体统计
    print(f"\n{'='*80}")
    print("总体统计")
    print(f"{'='*80}")
    print(f"  分析论文数: {len(results)}")
    print(f"  明确标注 arXiv 的引用总数: {total_declared}")
    print(f"  正确匹配数: {total_matched}")
    print(f"  漏连数: {total_missed}")
    print(f"  多余边数: {total_extra}")
    print(f"  召回率: {total_matched/total_declared*100:.1f}%" if total_declared > 0 else "N/A")

if __name__ == '__main__':
    main()

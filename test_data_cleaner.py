#!/usr/bin/env python3
"""
测试数据清洗 Agent
"""
import os
import sys
import re
sys.path.insert(0, '/home/QuanWei/DataCrawlerAgent')

from openai import OpenAI
from dotenv import load_dotenv
from agents.data_cleaner import DataCleanerAgent

load_dotenv('/home/QuanWei/DataCrawlerAgent/.env')

def test_cleaner():
    # 初始化 - 使用 .env 中的配置
    client = OpenAI(
        api_key=os.getenv("API_KEY"),
        base_url=os.getenv("MODEL_BASE_URL")
    )

    agent = DataCleanerAgent(
        client=client,
        system_prompt="你是一个学术论文格式专家，负责清洗和标准化论文文本。"
    )

    # 测试清洗单个论文
    paper_dir = '/home/QuanWei/DataCrawlerAgent/data/recent_papers/2410.10700'
    
    print("=" * 60)
    print("测试论文清洗: 2410.10700")
    print("=" * 60)
    
    result = agent.clean_single_paper(paper_dir, use_llm=True)
    
    print(f"\n状态: {result['status']}")
    print(f"清洗统计: {result['stats']}")
    
    if result['status'] == 'success':
        print("\n查看清洗后的 References 部分:")
        md_path = os.path.join(paper_dir, '2410.10700.md')
        with open(md_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 提取 References 部分
        ref_match = re.search(r'##\s*References\s*\n(.*?)(?:\n##|\Z)', content, re.DOTALL | re.IGNORECASE)
        if ref_match:
            refs = ref_match.group(1)[:2000]
            print(f"\n前 2000 字符:\n{refs}")
    
    return result

if __name__ == '__main__':
    test_cleaner()

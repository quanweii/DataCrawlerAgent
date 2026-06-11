"""
测试脚本 - 使用单篇论文测试LLM驱动的多智能体系统
"""
import os
import sys
import json
from dotenv import load_dotenv
from loguru import logger

# 加载环境变量
load_dotenv()

from orchestrator import DataCrawlerOrchestrator


def main():
    logger.remove()
    logger.add(sys.stderr, level="INFO",
               format="<green>{time:HH:mm:ss}</green> | <level>{level:<8}</level> | <cyan>{name}</cyan> - <level>{message}</level>")

    # 创建输出目录
    test_output = "./data/test_llm_pipeline"
    os.makedirs(test_output, exist_ok=True)
    os.makedirs(f"{test_output}/papers", exist_ok=True)
    os.makedirs(f"{test_output}/cleaned", exist_ok=True)
    os.makedirs(f"{test_output}/datasets", exist_ok=True)
    os.makedirs(f"{test_output}/graphs", exist_ok=True)

    # 创建协调器
    logger.info("创建LLM驱动的多智能体协调器...")
    orchestrator = DataCrawlerOrchestrator()

    # 使用一篇知名的 arXiv 论文测试
    # 2301.07041 - "Attention Is All You Need" (Transformer 论文)
    task = {
        "paper_sources": [
            {"type": "arxiv", "id": "2312.11805", "title": "Mamba: Linear-Time Sequence Modeling with Selective State Spaces"}
        ],
        "extract_multimodal": True,
        "output_dirs": {
            "papers": f"{test_output}/papers",
            "cleaned": f"{test_output}/cleaned",
            "datasets": f"{test_output}/datasets",
            "graphs": f"{test_output}/graphs"
        },
        "dataset_name": "test_llm_mamba",
        "output_format": "json",
        "device": "cpu"  # 使用CPU测试避免GPU问题
    }

    logger.info(f"测试任务: 下载并处理 arXiv:2312.11805 (Mamba)")

    # 执行LLM驱动的流水线
    result = orchestrator.execute_pipeline(task)

    # 打印结果
    print("\n" + "=" * 60)
    print("LLM驱动多智能体系统 - 测试结果")
    print("=" * 60)

    print(f"\n状态: {result['status']}")
    print(f"耗时: {result['duration_seconds']:.2f} 秒")

    if result.get("llm_report"):
        print(f"\n--- LLM协调器报告 ---")
        print(result["llm_report"])

    if result.get("stages"):
        print(f"\n--- 各阶段LLM评估 ---")
        for name, stage in result["stages"].items():
            eval_data = stage.get("_llm_evaluation", {})
            analysis = stage.get("_llm_analysis", {})
            print(f"\n  [{name}]")
            print(f"    状态: {stage.get('status', 'unknown')}")
            print(f"    LLM理解: {analysis.get('understanding', 'N/A')}")
            print(f"    自评质量: {eval_data.get('quality', 'N/A')}")
            if eval_data.get("issues"):
                print(f"    发现问题: {eval_data['issues']}")

    if result.get("errors"):
        print(f"\n错误: {result['errors']}")

    print("\n" + "=" * 60)

    return result


if __name__ == "__main__":
    main()

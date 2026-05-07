"""
数据爬取多Agent系统 - 主入口
Data Crawler Multi-Agent System

使用方法:
    python main.py

或通过配置文件:
    python main.py --config config.json
"""
import os
import sys
import json
import argparse
from datetime import datetime
from dotenv import load_dotenv
from loguru import logger

from orchestrator import DataCrawlerOrchestrator
from orchestrator_parallel import ParallelDataCrawlerOrchestrator


def setup_logging(log_level: str = "INFO"):
    """配置日志系统"""
    logger.remove()
    logger.add(
        sys.stderr,
        level=log_level,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>"
    )
    logger.add(
        "logs/data_crawler_{time}.log",
        level="DEBUG",
        rotation="10 MB",
        retention="7 days",
        encoding="utf-8"
    )


def load_task_config(config_path: str) -> dict:
    """从配置文件加载任务配置"""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"配置文件不存在: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


def create_sample_task() -> dict:
    """创建示例任务配置"""
    return {
        "paper_sources": [
            {"type": "arxiv", "id": "2301.07041", "title": "示例论文1"},
            {"type": "arxiv", "id": "2301.00001", "title": "示例论文2"},
            # {"type": "doi", "id": "10.1234/example.2023.001", "title": "示例DOI论文"}
        ],
        "extract_multimodal": True,
        "cleaning_rules": {
            "min_text_length": 50,
            "remove_special_chars": True,
            "normalize_whitespace": True
        },
        "dataset_name": f"multimodal_dataset_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        "output_format": "json",
        "output_dirs": {
            "papers": "./data/papers",
            "cleaned": "./data/cleaned",
            "datasets": "./data/datasets"
        },
        "include_metadata": True
    }


def print_result_summary(result: dict):
    """打印执行结果摘要"""
    print("\n" + "=" * 60)
    print("📊 执行结果摘要")
    print("=" * 60)

    print(f"\n状态: {'✅ 成功' if result['status'] == 'success' else '❌ 失败'}")
    print(f"开始时间: {result['start_time']}")
    print(f"结束时间: {result['end_time']}")
    print(f"总耗时: {result['duration_seconds']:.2f} 秒")

    if result['status'] == 'success':
        final = result.get('final_output', {})
        print(f"\n📁 最终输出:")
        print(f"  - 数据集路径: {final.get('dataset_path', 'N/A')}")
        print(f"  - 节点数量: {final.get('node_count', 0)}")
        print(f"  - 边数量: {final.get('edge_count', 0)}")
        print(f"  - 模态分布: {final.get('modality_distribution', {})}")

        print("\n📈 各阶段统计:")
        stages = result.get('stages', {})

        stage1 = stages.get('paper_collection', {})
        print(f"  [论文收集] 论文数: {len(stage1.get('papers', []))}, 多模态数据: {len(stage1.get('multimodal_data', []))}")

        stage2 = stages.get('data_cleaning', {})
        stats2 = stage2.get('stats', {})
        print(f"  [数据清洗] 输入: {stats2.get('total_input', 0)}, 输出: {stats2.get('total_output', 0)}, 去重: {stats2.get('duplicates_removed', 0)}")

        stage3 = stages.get('dataset_formatting', {})
        stats3 = stage3.get('stats', {})
        print(f"  [数据封装] 节点: {stats3.get('node_count', 0)}, 边: {stats3.get('edge_count', 0)}")

    if result.get('errors'):
        print(f"\n⚠️ 错误信息:")
        for error in result['errors']:
            print(f"  - {error}")

    print("\n" + "=" * 60)


def interactive_mode(orchestrator: DataCrawlerOrchestrator):
    """交互式模式"""
    print("\n🤖 数据爬取多Agent系统 - 交互模式")
    print("输入 'help' 查看可用命令, 'quit' 退出\n")

    while True:
        try:
            user_input = input(">>> ").strip()

            if not user_input:
                continue

            if user_input.lower() in ['quit', 'exit', 'q']:
                print("再见!")
                break

            elif user_input.lower() == 'help':
                print("""
可用命令:
  run       - 运行完整的处理流水线
  config    - 显示当前配置
  agents    - 列出所有Agent
  tools     - 列出所有MCP工具
  history   - 显示执行历史
  status    - 显示当前状态
  help      - 显示帮助信息
  quit      - 退出程序
                """)

            elif user_input.lower() == 'run':
                task = create_sample_task()
                print("\n请输入论文来源（每行一个，格式: type id title）")
                print("示例: arxiv 2301.07041 示例论文")
                print("输入空行结束")

                sources = []
                while True:
                    line = input("  > ").strip()
                    if not line:
                        break
                    parts = line.split(maxsplit=2)
                    if len(parts) >= 2:
                        source = {"type": parts[0], "id": parts[1]}
                        if len(parts) == 3:
                            source["title"] = parts[2]
                        sources.append(source)

                if sources:
                    task["paper_sources"] = sources

                result = orchestrator.execute_pipeline(task)
                print_result_summary(result)

            elif user_input.lower() == 'config':
                print("\n当前配置:")
                print(f"  API_KEY: {'*' * 8}{os.getenv('API_KEY', '')[-4:] if os.getenv('API_KEY') else 'Not Set'}")
                print(f"  MODEL_BASE_URL: {os.getenv('MODEL_BASE_URL', 'Not Set')}")
                print(f"  MODEL_NAME: {os.getenv('MODEL_NAME', 'qwen-max')}")

            elif user_input.lower() == 'agents':
                print("\n已注册的Agent:")
                for name, agent in orchestrator.agents.items():
                    print(f"  - {name}: {agent.agent_name}")

            elif user_input.lower() == 'tools':
                print("\n已注册的MCP工具:")
                tools = orchestrator.mcp_tool_manager.list_tools()
                for name, schema in tools.items():
                    print(f"  - {name}: {schema.get('description', 'No description')}")

            elif user_input.lower() == 'history':
                print("\n执行历史:")
                history = orchestrator.get_execution_history()
                for i, record in enumerate(history[-5:], 1):
                    print(f"  {i}. {record['start_time']} - {record['status']} ({record['duration_seconds']:.2f}s)")

            elif user_input.lower() == 'status':
                print(f"\n当前状态: {orchestrator.get_current_status()}")

            else:
                print(f"未知命令: {user_input}. 输入 'help' 查看可用命令")

        except KeyboardInterrupt:
            print("\n\n操作已取消")
            break
        except Exception as e:
            logger.error(f"执行错误: {e}")
            print(f"错误: {e}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="数据爬取多Agent系统")
    parser.add_argument(
        "--config", "-c",
        type=str,
        default=None,
        help="任务配置文件路径 (JSON格式)"
    )
    parser.add_argument(
        "--interactive", "-i",
        action="store_true",
        help="启动交互模式"
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="日志级别"
    )
    parser.add_argument(
        "--sample",
        action="store_true",
        help="使用示例配置运行"
    )
    parser.add_argument(
        "--batch", "-b",
        action="store_true",
        help="批量下载模式（从arxiv元数据筛选论文）"
    )
    parser.add_argument(
        "--num-gpus",
        type=int,
        default=8,
        help="并行提取使用的GPU数量 (默认: 8)"
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=8,
        help="CPU并行worker数，用于清洗和引用图构建 (默认: 8)"
    )
    parser.add_argument(
        "--sequential",
        action="store_true",
        help="禁用并行，使用顺序执行（调试用）"
    )

    args = parser.parse_args()

    # 加载环境变量
    load_dotenv()

    # 配置日志
    setup_logging(args.log_level)

    # 创建必要的目录
    os.makedirs("logs", exist_ok=True)
    os.makedirs("data/papers", exist_ok=True)
    os.makedirs("data/cleaned", exist_ok=True)
    os.makedirs("data/datasets", exist_ok=True)

    # 检查环境变量
    if not os.getenv("API_KEY"):
        logger.warning("未设置 API_KEY 环境变量，请在 .env 文件中配置")

    # 创建主协调器
    if not args.sequential and args.batch:
        orchestrator = ParallelDataCrawlerOrchestrator(
            num_gpus=args.num_gpus,
            num_cpu_workers=args.num_workers
        )
        logger.info(f"使用并行模式: {args.num_gpus} GPUs, {args.num_workers} CPU workers")
    else:
        orchestrator = DataCrawlerOrchestrator()
        if args.sequential:
            logger.info("使用顺序模式（调试）")

    if args.interactive:
        # 交互模式
        interactive_mode(orchestrator)

    elif args.sample:
        # 使用示例配置运行
        task = create_sample_task()
        result = orchestrator.execute_pipeline(task)
        print_result_summary(result)

    elif args.batch:
        # 批量下载模式
        if args.config:
            task = load_task_config(args.config)
        else:
            # 默认批量下载配置
            task = {
                "metadata_path": "/home/QuanWei/arxiv-metadata-oai-snapshot.json",
                "category": "cs.AI",
                "limit": 20,
                "extract_multimodal": True,
                "device": "cuda:0",
                "output_dirs": {
                    "papers": "./data/cs_ai_papers",
                    "cleaned": "./data/cs_ai_cleaned",
                    "datasets": "./data/cs_ai_datasets"
                },
                "dataset_name": f"cs_ai_dataset_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                "output_format": "json"
            }
        result = orchestrator.execute_batch_pipeline(task)
        print_result_summary(result)

    elif args.config:
        # 从配置文件运行
        task = load_task_config(args.config)
        result = orchestrator.execute_pipeline(task)
        print_result_summary(result)

    else:
        # 默认：显示帮助并启动交互模式
        print("""
╔══════════════════════════════════════════════════════════════╗
║           数据爬取多Agent系统 (Data Crawler Agent)            ║
╠══════════════════════════════════════════════════════════════╣
║  使用方法:                                                    ║
║    python main.py --interactive    # 交互模式                 ║
║    python main.py --sample         # 示例运行                 ║
║    python main.py --config xxx.json # 配置文件运行            ║
║    python main.py --batch          # 批量下载cs.AI论文        ║
║    python main.py --batch -c cfg.json # 批量下载+配置文件     ║
╚══════════════════════════════════════════════════════════════╝
        """)
        interactive_mode(orchestrator)


if __name__ == "__main__":
    main()

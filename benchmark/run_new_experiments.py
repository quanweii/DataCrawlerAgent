"""
运行新增的三个补充实验
1. Few-shot学习曲线 (1/3/5/10/20-shot)
2. 跨域链接预测
3. 训练集比例学习曲线

用法:
    cd /home/dev_qw/DataCrawlerAgent
    python -m benchmark.run_new_experiments --device cuda:0
"""
import os
import sys
import argparse
import torch
from loguru import logger

logger.add(sys.stderr, level="INFO")


def main():
    parser = argparse.ArgumentParser(description="运行AMMAG新增补充实验")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--exp", type=str, default="all",
                        choices=["all", "fewshot_curve", "cross_domain_lp", "learning_curve"])
    args = parser.parse_args()

    # 加载数据集
    from benchmark.data.dataset import AMMAGDataset
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    if args.exp in ["fewshot_curve", "all"]:
        logger.info("=" * 60)
        logger.info("实验1: Few-shot学习曲线 (1/3/5/10/20-shot)")
        logger.info("=" * 60)
        dataset = AMMAGDataset(root=root, modality="text+image+formula")
        from benchmark.tasks.few_shot_curve import run_few_shot_curve
        run_few_shot_curve(dataset, device=args.device)
        del dataset
        torch.cuda.empty_cache()

    if args.exp in ["cross_domain_lp", "all"]:
        logger.info("=" * 60)
        logger.info("实验2: 跨域链接预测")
        logger.info("=" * 60)
        for modality in ["text", "text+image+formula", "image+formula"]:
            dataset = AMMAGDataset(root=root, modality=modality)
            from benchmark.tasks.cross_domain_link_prediction import run_cross_domain_link_prediction
            run_cross_domain_link_prediction(dataset, device=args.device)
            del dataset
            torch.cuda.empty_cache()

    if args.exp in ["learning_curve", "all"]:
        logger.info("=" * 60)
        logger.info("实验3: 训练集比例学习曲线")
        logger.info("=" * 60)
        dataset = AMMAGDataset(root=root, modality="text+image+formula")
        from benchmark.tasks.learning_curve import run_multi_model_learning_curve
        run_multi_model_learning_curve(dataset, device=args.device)
        del dataset
        torch.cuda.empty_cache()

    logger.info("所有新增实验完成!")


if __name__ == "__main__":
    main()

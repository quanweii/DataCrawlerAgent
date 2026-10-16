#!/usr/bin/env python3
"""
Reddit 数据准备脚本
从 Zenodo 的 Pushshift 数据集中下载并过滤 Reddit 数据

使用方法:
    conda activate agent_env
    python scripts/run_reddit_data_prep.py

数据来源: Zenodo Record 3608135 (The Pushshift Reddit Dataset)
    RS_2019-04.zst: 2019年4月所有 Reddit submissions
    RC_2019-04.zst: 2019年4月所有 Reddit comments
"""
import os
import sys
import json
import time

# 添加项目根目录到 sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.pushshift_loader import PushshiftDataLoader
from loguru import logger


# 目标 subreddit - 覆盖 AI/ML/数据科学/科技/科学 多个方向
# 选择标准：活跃度高、帖子质量好、彼此有区分度（用于 node classification）
TARGET_SUBREDDITS = [
    "MachineLearning",       # 机器学习研究
    "datascience",           # 数据科学实践
    "artificial",            # AI 新闻和讨论
    "deeplearning",          # 深度学习技术
    "learnmachinelearning",  # ML 学习
    "statistics",            # 统计学
    "computerscience",       # 计算机科学
    "programming",           # 编程
    "technology",            # 科技新闻
    "science",               # 科学
]

POSTS_PER_SUBREDDIT = 1500   # 每个 subreddit 收集 1500 帖子 → 总计 ~15K
COMMENTS_PER_POST = 25        # 每帖最多 25 评论
DATA_DIR = "./data/reddit_raw"


def download_from_zenodo():
    """从 Zenodo 下载 Pushshift 数据文件（使用 curl 避免 SSL 问题）"""

    # 已知的 Zenodo 文件 URL（Record #3608135）
    ZENODO_FILES = {
        "RS_2019-04.zst": {
            "url": "https://zenodo.org/api/records/3608135/files/RS_2019-04.zst/content",
            "size_gb": 5.2,
            "desc": "submissions",
        },
        "RC_2019-04.zst": {
            "url": "https://zenodo.org/api/records/3608135/files/RC_2019-04.zst/content",
            "size_gb": 14.5,
            "desc": "comments",
        },
    }

    os.makedirs(DATA_DIR, exist_ok=True)

    downloaded = {}
    for filename, info in ZENODO_FILES.items():
        target_path = os.path.join(DATA_DIR, filename)

        if os.path.exists(target_path) and os.path.getsize(target_path) > 100_000_000:
            size_mb = os.path.getsize(target_path) / (1024**2)
            logger.info(f"  跳过已存在的文件: {filename} ({size_mb:.0f} MB)")
            downloaded[filename] = target_path
            continue

        logger.info(
            f"  开始下载: {filename} ({info['desc']}, ~{info['size_gb']:.1f} GB)"
        )
        logger.info(f"  目标: {target_path}")

        # 使用 curl 下载（支持断点续传，忽略 SSL 验证问题）
        ret = os.system(
            f'curl -L -C - --insecure -o "{target_path}" "{info["url"]}"'
        )
        if ret != 0:
            logger.warning(f"  下载失败 (ret={ret}): {filename}")
            # 尝试不用断点续传
            ret = os.system(
                f'curl -L --insecure -o "{target_path}" "{info["url"]}"'
            )

        if os.path.exists(target_path):
            size_mb = os.path.getsize(target_path) / (1024**2)
            logger.info(f"  下载完成: {filename} ({size_mb:.0f} MB)")
            downloaded[filename] = target_path
        else:
            logger.error(f"  下载失败: {filename}")

    return downloaded


def main():
    logger.info("=" * 60)
    logger.info("Reddit 数据准备: Pushshift → 过滤 → JSON")
    logger.info("=" * 60)

    # 检查是否已有预计算数据
    loader = PushshiftDataLoader(data_dir=DATA_DIR)
    if loader.load_precomputed():
        stats = {
            "total_posts": len(loader.get_posts()),
            "total_comments": sum(
                len(c) for c in loader.get_comments().values()
            ),
        }
        logger.info(f"已有预计算数据: {stats}")
        if stats["total_posts"] >= 5000:
            logger.info("数据量足够，跳过重新加载")
            return

    # 1. 下载数据
    logger.info("\n--- Step 1: 下载 Pushshift 数据 ---")
    download_map = download_from_zenodo()

    # 2. 过滤数据
    logger.info("\n--- Step 2: 过滤 Reddit 数据 ---")

    # 查找下载的数据文件
    submissions_path = download_map.get("RS_2019-04.zst")
    comments_path = download_map.get("RC_2019-04.zst")

    # 也检查其他可能的位置
    if not submissions_path or not os.path.exists(submissions_path):
        for name in ["RS_2019-04.zst", "submissions.zst", "submissions.json.gz", "submissions.json"]:
            p = os.path.join(DATA_DIR, name)
            if os.path.exists(p):
                submissions_path = p
                break

    if not comments_path or not os.path.exists(comments_path):
        for name in ["RC_2019-04.zst", "comments.zst", "comments.json.gz", "comments.json"]:
            p = os.path.join(DATA_DIR, name)
            if os.path.exists(p):
                comments_path = p
                break

    logger.info(f"  Submissions: {submissions_path}")
    logger.info(f"  Comments: {comments_path}")

    stats = loader.load_from_file(
        submissions_path=submissions_path,
        comments_path=comments_path,
        subreddits=TARGET_SUBREDDITS,
        posts_per_subreddit=POSTS_PER_SUBREDDIT,
        comments_per_post=COMMENTS_PER_POST,
    )

    logger.info(f"\n最终统计:")
    logger.info(f"  帖子数: {stats['total_posts']}")
    logger.info(f"  评论数: {stats['total_comments']}")
    logger.info(f"  唯一用户: {stats['unique_authors']}")
    logger.info(f"  Subreddit 分布: {json.dumps(stats['subreddit_distribution'], indent=2)}")

    # 3. 保存用户数据
    users = loader.get_users()
    users_path = os.path.join(DATA_DIR, "users.json")
    with open(users_path, 'w', encoding='utf-8') as f:
        json.dump(users, f, ensure_ascii=False, indent=2)
    logger.info(f"  用户数据: {len(users)} 用户 → {users_path}")

    logger.info("\n✅ Reddit 数据准备完成!")
    return stats


if __name__ == "__main__":
    main()

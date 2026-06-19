#!/bin/bash
# Reddit 数据自动处理监控脚本
# 持续监控 Zenodo 下载进度，当数据足够时自动处理并启动 pipeline
#
# 用法: bash scripts/monitor_and_run_reddit.sh

set -e
cd "$(dirname "$0")/.."

MIN_POSTS=2000      # 最少需要的帖子数
MIN_FILE_SIZE_MB=300  # 最少需要的文件大小 (MB)
CHECK_INTERVAL=120    # 检查间隔 (秒)
MAX_WAIT=7200         # 最大等待时间 (秒) = 2 小时

RAW_DIR="./data/reddit_raw"
RS_FILE="$RAW_DIR/RS_2019-04.zst"
LOG_PREFIX="[RedditMonitor]"

echo "$LOG_PREFIX 开始监控 Pushshift 下载..."
echo "$LOG_PREFIX 目标: 至少 $MIN_POSTS 帖子 或 $MIN_FILE_SIZE_MB MB"
echo "$LOG_PREFIX 检查间隔: ${CHECK_INTERVAL}s"

elapsed=0
while [ $elapsed -lt $MAX_WAIT ]; do
    if [ -f "$RS_FILE" ]; then
        file_size_mb=$(du -m "$RS_FILE" 2>/dev/null | cut -f1)

        # 尝试处理当前数据
        post_count=$(conda run -n agent_env python3 -c "
import json, sys, os
sys.path.insert(0, '.')
from tools.pushshift_loader import PushshiftDataLoader

fsize = os.path.getsize('$RS_FILE')
loader = PushshiftDataLoader(data_dir='$RAW_DIR')

# Check if we already have enough precomputed data
if loader.load_precomputed():
    print(len(loader.get_posts()))
    sys.exit(0)

stats = loader.load_from_file(
    submissions_path='$RS_FILE',
    comments_path=None,
    subreddits=['MachineLearning','datascience','artificial','deeplearning',
                'learnmachinelearning','statistics','computerscience',
                'programming','technology','science'],
    posts_per_subreddit=1500,
)
print(stats['total_posts'])
" 2>/dev/null | tail -1)

        echo "$LOG_PREFIX [${elapsed}s] 文件: ${file_size_mb}MB, 帖子: ${post_count:-0}"

        if [ "${post_count:-0}" -ge "$MIN_POSTS" ] 2>/dev/null; then
            echo "$LOG_PREFIX ✅ 数据足够! (${post_count} 帖子 >= ${MIN_POSTS})"
            break
        fi

        if [ "${file_size_mb:-0}" -ge "$MIN_FILE_SIZE_MB" ] 2>/dev/null; then
            echo "$LOG_PREFIX ✅ 文件大小足够 (${file_size_mb}MB >= ${MIN_FILE_SIZE_MB}MB)"
            # 处理最终数据
            break
        fi
    else
        echo "$LOG_PREFIX [${elapsed}s] 等待文件创建..."
    fi

    sleep $CHECK_INTERVAL
    elapsed=$((elapsed + CHECK_INTERVAL))
done

echo "$LOG_PREFIX 开始最终数据处理..."

# 最终数据处理（含评论加载）
conda run -n agent_env python3 -c "
import json, sys, os
sys.path.insert(0, '.')
from tools.pushshift_loader import PushshiftDataLoader

loader = PushshiftDataLoader(data_dir='$RAW_DIR')

# Load precomputed or process
if not loader.load_precomputed():
    stats = loader.load_from_file(
        submissions_path='$RAW_DIR/RS_2019-04.zst',
        comments_path=None,
        subreddits=['MachineLearning','datascience','artificial','deeplearning',
                    'learnmachinelearning','statistics','computerscience',
                    'programming','technology','science'],
        posts_per_subreddit=1500,
    )
    print(f'Processed: {stats}')
else:
    posts = loader.get_posts()
    comments = loader.get_comments()
    total_c = sum(len(c) for c in comments.values())
    print(f'Precomputed: {len(posts)} posts, {total_c} comments')

# Save users
users = loader.get_users()
with open('$RAW_DIR/users.json', 'w') as f:
    json.dump(users, f, ensure_ascii=False, indent=2)
print(f'{len(users)} users saved')
" 2>&1

echo "$LOG_PREFIX 运行 Reddit pipeline..."
conda run -n agent_env python main.py \
    --domain reddit \
    --config reddit_config.json \
    --log-level INFO \
    2>&1 | tee "logs/reddit_pipeline_$(date '+%Y%m%d_%H%M%S').log"

echo "$LOG_PREFIX ✅ Reddit pipeline 完成!"

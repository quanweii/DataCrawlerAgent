#!/bin/bash
# Reddit 数据集完整构建流水线
# 使用多智能体系统从 Pushshift 数据构建 Reddit 多模态属性图
#
# 流程:
#   1. 从 Zenodo 下载 Pushshift 数据 (如尚未下载)
#   2. 过滤目标 subreddit 的帖子和评论
#   3. 运行多智能体 pipeline 构建图
#
# 用法:
#   conda activate agent_env
#   bash scripts/run_reddit_pipeline.sh

set -e
cd "$(dirname "$0")/.."

echo "============================================================"
echo "  Reddit 多模态属性图数据集构建"
echo "  数据源: Pushshift Reddit Dataset (Zenodo #3608135)"
echo "  时间: $(date '+%Y-%m-%d %H:%M:%S')"
echo "============================================================"

RAW_DIR="./data/reddit_raw"
mkdir -p "$RAW_DIR"

# Step 1: 准备数据（下载 + 过滤）
echo ""
echo "--- Step 1: 准备 Pushshift 数据 ---"

# 检查是否已有预计算数据
if [ -f "$RAW_DIR/posts.json" ]; then
    POST_COUNT=$(python3 -c "import json; print(len(json.load(open('$RAW_DIR/posts.json'))))" 2>/dev/null || echo "0")
    echo "📊 已有预计算帖子数据: $POST_COUNT 个"
    if [ "$POST_COUNT" -ge 5000 ]; then
        echo "✅ 数据量足够，跳过数据准备"
    else
        echo "⚠️  数据量不足 ($POST_COUNT < 5000)，重新准备..."
        conda run -n agent_env python scripts/run_reddit_data_prep.py
    fi
else
    echo "📥 未找到预计算数据，开始下载和过滤..."
    conda run -n agent_env python scripts/run_reddit_data_prep.py
fi

# Step 2: 验证数据
echo ""
echo "--- Step 2: 验证数据 ---"

if [ ! -f "$RAW_DIR/posts.json" ]; then
    echo "❌ 数据文件不存在: $RAW_DIR/posts.json"
    echo "   Pushshift 数据可能尚未下载完成。"
    echo "   请等待下载完成后重新运行此脚本。"
    exit 1
fi

POST_COUNT=$(python3 -c "import json; print(len(json.load(open('$RAW_DIR/posts.json'))))" 2>/dev/null || echo "0")
COMMENT_COUNT=$(python3 -c "import json; d=json.load(open('$RAW_DIR/comments.json')); print(sum(len(v) for v in d.values()))" 2>/dev/null || echo "0")
echo "📊 帖子: $POST_COUNT, 评论: $COMMENT_COUNT"

if [ "$POST_COUNT" -lt 100 ]; then
    echo "❌ 数据量太少 ($POST_COUNT < 100)，无法构建有意义的图"
    exit 1
fi

# Step 3: 运行 pipeline
echo ""
echo "🚀 启动 Reddit 多智能体 pipeline..."
echo "   Config: reddit_config.json"
echo "   Domain: reddit"
echo "   Graph mode: heterogeneous"
echo "   Data source: file (Pushshift)"
echo ""

conda run -n agent_env python main.py \
    --domain reddit \
    --config reddit_config.json \
    --log-level INFO \
    2>&1 | tee "logs/reddit_pipeline_$(date '+%Y%m%d_%H%M%S').log"

EXIT_CODE=${PIPESTATUS[0]}

echo ""
echo "============================================================"
if [ $EXIT_CODE -eq 0 ]; then
    echo "  ✅ Reddit pipeline 完成!"
else
    echo "  ❌ Reddit pipeline 失败 (exit code: $EXIT_CODE)"
fi
echo "  时间: $(date '+%Y-%m-%d %H:%M:%S')"
echo "============================================================"

exit $EXIT_CODE

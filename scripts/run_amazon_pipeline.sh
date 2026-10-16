#!/bin/bash
# Amazon 数据集完整构建流水线
# 使用多智能体系统从 McAuley 数据集构建 Amazon 多模态属性图
#
# 用法:
#   conda activate agent_env
#   bash scripts/run_amazon_pipeline.sh

set -e
cd "$(dirname "$0")/.."

echo "============================================================"
echo "  Amazon 多模态属性图数据集构建"
echo "  数据源: McAuley Amazon Product Dataset (2018)"
echo "  时间: $(date '+%Y-%m-%d %H:%M:%S')"
echo "============================================================"

# 确认数据文件存在
RAW_DIR="./data/amazon_raw"
if [ ! -d "$RAW_DIR" ]; then
    echo "❌ 数据目录不存在: $RAW_DIR"
    exit 1
fi

META_COUNT=$(ls -1 "$RAW_DIR"/meta_*.json.gz 2>/dev/null | wc -l)
echo "📊 元数据文件: $META_COUNT 个"

if [ "$META_COUNT" -lt 3 ]; then
    echo "⚠️  元数据文件不足 (需要3个)，请先下载数据"
    exit 1
fi

# 运行 pipeline
echo ""
echo "🚀 启动 Amazon 多智能体 pipeline..."
echo "   Config: amazon_config.json"
echo "   Domain: amazon"
echo "   Graph mode: heterogeneous"
echo ""

conda run -n agent_env python main.py \
    --domain amazon \
    --config amazon_config.json \
    --log-level INFO \
    2>&1 | tee "logs/amazon_pipeline_$(date '+%Y%m%d_%H%M%S').log"

EXIT_CODE=${PIPESTATUS[0]}

echo ""
echo "============================================================"
if [ $EXIT_CODE -eq 0 ]; then
    echo "  ✅ Amazon pipeline 完成!"
else
    echo "  ❌ Amazon pipeline 失败 (exit code: $EXIT_CODE)"
fi
echo "  时间: $(date '+%Y-%m-%d %H:%M:%S')"
echo "============================================================"

exit $EXIT_CODE

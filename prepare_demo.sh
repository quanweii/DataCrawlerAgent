#!/bin/bash
# ============================================================
# 演示准备脚本 - 为5篇精选论文准备 demo 数据
# 运行方式: bash prepare_demo.sh
# 在录制视频前执行，确保所有数据已就绪
# ============================================================
set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PROJECT_ROOT
SRC_DIR="$PROJECT_ROOT/data/cs_si_papers"
DST_DIR="$PROJECT_ROOT/data/demo_papers"
EDGE_FILE="$PROJECT_ROOT/data/citation_edges_full/cs_si_citation_edges_full.csv"

PAPERS=(
    "2206.00606"
    "2409.15369"
    "2205.14651"
    "2202.13491"
    "2203.03602"
)

echo "============================================"
echo "  AMMAG 演示数据准备"
echo "============================================"
echo ""

# Step 1: Copy extracted paper data (md, images, npy)
echo "[1/4] 复制论文多模态提取数据..."
mkdir -p "$DST_DIR"
total_md=0
total_img=0
total_formula=0

for paper_id in "${PAPERS[@]}"; do
    src="$SRC_DIR/$paper_id"
    dst="$DST_DIR/$paper_id"

    if [ -d "$src" ]; then
        mkdir -p "$dst"
        # Copy markdown
        [ -f "$src/$paper_id.md" ] && cp "$src/$paper_id.md" "$dst/" && total_md=$((total_md+1))
        # Copy npy features
        [ -f "$src/$paper_id.npy" ] && cp "$src/$paper_id.npy" "$dst/"
        # Copy images
        if [ -d "$src/images" ]; then
            img_count=$(ls "$src/images" | wc -l)
            cp -r "$src/images" "$dst/" 2>/dev/null || true
            total_img=$((total_img + img_count))
        fi
        # Count formulas in md
        if [ -f "$dst/$paper_id.md" ]; then
            fc=$(grep -c '\$\$' "$dst/$paper_id.md" 2>/dev/null || echo 0)
            total_formula=$((total_formula + fc))
        fi
        echo "  ✓ $paper_id"
    else
        echo "  ✗ $paper_id: 源数据不存在"
    fi
done

echo ""
echo "  复制完成: $total_md 篇MD, $total_img 张图片, ~$total_formula 条公式"

# Step 2: Extract citation edges for these 5 papers
echo ""
echo "[2/4] 提取引用边数据..."
EDGE_DST="$PROJECT_ROOT/data/demo_graphs"
mkdir -p "$EDGE_DST"

# Extract edges where source is one of our papers
python3 << 'PYEOF'
import os

SRC = os.path.join(os.environ["PROJECT_ROOT"], "data/citation_edges_full/cs_si_citation_edges_full.csv")
DST = os.path.join(os.environ["PROJECT_ROOT"], "data/demo_graphs/demo_citation_edges.csv")
PAPERS = {"2206.00606","2409.15369","2205.14651","2202.13491","2203.03602"}

total = 0
with open(SRC) as fin, open(DST, 'w') as fout:
    fout.write("source_id,target_id,source_title,target_title,reference_text\n")
    for line in fin:
        src_id = line.split(',', 1)[0]
        if src_id in PAPERS:
            fout.write(line)
            total += 1

print(f"  总计: {total} 条引用边")
PYEOF

# Step 3: Summary
echo ""
echo "[3/4] 演示数据概览:"
python3 << 'PYEOF'
import os

DST = os.path.join(os.environ["PROJECT_ROOT"], "data/demo_papers")
EDGE = os.path.join(os.environ["PROJECT_ROOT"], "data/demo_graphs/demo_citation_edges.csv")
PAPERS = ["2206.00606","2409.15369","2205.14651","2202.13491","2203.03602"]

print("  ┌──────────────────────────────────────────────────────┐")
for p in PAPERS:
    md_size = img_count = fc = ec = 0
    md_path = os.path.join(DST, p, f"{p}.md")
    img_dir = os.path.join(DST, p, "images")
    if os.path.exists(md_path):
        md_size = os.path.getsize(md_path) // 1024
        with open(md_path) as f:
            fc = f.read().count("$$")
    if os.path.isdir(img_dir):
        img_count = len([f for f in os.listdir(img_dir) if os.path.isfile(os.path.join(img_dir, f))])
    if os.path.exists(EDGE):
        with open(EDGE) as f:
            ec = sum(1 for line in f if line.startswith(p + ","))
    print(f"  │ {p:10s}  MD: {md_size:4d}KB  图片: {img_count:3d}  公式: {fc:3d}  引用边: {ec:3d} │")
print("  └──────────────────────────────────────────────────────┘")
PYEOF

# Step 4: Prepare cleaned data directory
echo ""
echo "[4/4] 创建输出目录..."
mkdir -p $PROJECT_ROOT/data/demo_cleaned
mkdir -p $PROJECT_ROOT/data/demo_datasets

echo ""
echo "============================================"
echo "  准备完成！可以开始录制演示视频"
echo "============================================"
echo ""
echo "  运行命令: python main.py --config demo_config.json"
echo ""
echo "  关键文件位置（录制时展示）:"
echo "  - 配置文件: demo_config.json"
echo "  - 提取的MD文本: data/demo_papers/{id}/{id}.md"
echo "  - 提取的图片: data/demo_papers/{id}/images/"
echo "  - 提取的特征: data/demo_papers/{id}/{id}.npy"
echo "  - 引用边CSV: data/demo_graphs/demo_citation_edges.csv"
echo "  - 最终数据集: data/demo_datasets/demo_multimodal_graph.json"

#!/usr/bin/env python3
"""
仅编码Figure图片（排除Table），生成figure-only image embeddings
"""
import os, sys, re, glob, json, torch, numpy as np
from collections import defaultdict
from PIL import Image
from loguru import logger
from transformers import CLIPModel, CLIPProcessor

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

DATA_ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
PROCESSED_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "benchmark", "processed")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

DOMAINS = {
    'cs_si': 'cs_si_papers',
    'cs_lo': 'cs_lo_papers',
    'math_lo': 'math_lo_papers',
    'physics_socph': 'physics_socph_papers',
}


def get_figure_indices(md_path):
    """从markdown中提取figure图片的索引列表"""
    with open(md_path, 'r', errors='ignore') as f:
        content = f.read()
    figs = re.findall(r'!\[figure\s+(\d+)\]', content, re.IGNORECASE)
    return [int(x) for x in figs]


def find_image(paper_path, domain_dir, pid, idx):
    """查找图片文件"""
    for candidate in [
        os.path.join(paper_path, 'images', f'{idx}.png'),
        os.path.join(domain_dir, f'{pid}_images', f'{idx}.png'),
    ]:
        if os.path.exists(candidate):
            return candidate
    return None


def main():
    logger.info("Loading CLIP model...")
    model = CLIPModel.from_pretrained('openai/clip-vit-base-patch32').to(DEVICE)
    processor = CLIPProcessor.from_pretrained('openai/clip-vit-base-patch32')
    model.eval()

    # Load paper_ids from graph
    graph = torch.load(os.path.join(PROCESSED_DIR, 'graph_data.pt'), map_location='cpu')
    all_paper_ids = list(graph['paper_ids'])
    pid_set = set(all_paper_ids)
    logger.info(f"Total papers in graph: {len(all_paper_ids)}")

    # Build pid -> figure indices mapping
    pid_to_fig_indices = {}
    stats = defaultdict(lambda: {'total': 0, 'has_fig': 0, 'fig_count': 0, 'tbl_count': 0})

    for dk, dd in DOMAINS.items():
        domain_dir = os.path.join(DATA_ROOT, dd)
        for paper_path in sorted(glob.glob(os.path.join(domain_dir, '*'))):
            if not os.path.isdir(paper_path):
                continue
            pid = os.path.basename(paper_path)
            if pid not in pid_set:
                continue
            stats[dk]['total'] += 1

            md_path = os.path.join(paper_path, f'{pid}.md')
            if not os.path.exists(md_path):
                pid_to_fig_indices[pid] = []
                continue

            fig_indices = get_figure_indices(md_path)
            pid_to_fig_indices[pid] = fig_indices

            if fig_indices:
                stats[dk]['has_fig'] += 1
            stats[dk]['fig_count'] += len(fig_indices)

            # Also count tables for stats
            with open(md_path, 'r', errors='ignore') as f:
                content = f.read()
            tbl_count = len(re.findall(r'!\[table\s+\d+\]', content, re.IGNORECASE))
            stats[dk]['tbl_count'] += tbl_count

    logger.info("Figure index statistics:")
    for dk in DOMAINS:
        s = stats[dk]
        logger.info(f"  {dk}: {s['total']} papers, {s['has_fig']} have figures "
                     f"({s['has_fig']/s['total']*100:.1f}%), "
                     f"total figures={s['fig_count']}, tables={s['tbl_count']}")

    # Encode figure-only images
    logger.info("Encoding figure-only images...")
    embeddings = {}
    pids_in_order = []

    batch_size = 32
    batch_images = []
    batch_pids = []

    for dk, dd in DOMAINS.items():
        domain_dir = os.path.join(DATA_ROOT, dd)
        for paper_path in sorted(glob.glob(os.path.join(domain_dir, '*'))):
            if not os.path.isdir(paper_path):
                continue
            pid = os.path.basename(paper_path)
            if pid not in pid_set:
                continue

            pids_in_order.append(pid)
            fig_indices = pid_to_fig_indices.get(pid, [])

            if not fig_indices:
                embeddings[pid] = None
                continue

            # Load figure images
            paper_images = []
            for idx in fig_indices:
                img_file = find_image(paper_path, domain_dir, pid, idx)
                if img_file:
                    try:
                        img = Image.open(img_file).convert('RGB')
                        paper_images.append(img)
                    except:
                        pass

            if not paper_images:
                embeddings[pid] = None
                continue

            # Encode all figures for this paper, then mean-pool
            with torch.no_grad():
                all_embs = []
                for i in range(0, len(paper_images), batch_size):
                    batch = paper_images[i:i+batch_size]
                    inputs = processor(images=batch, return_tensors="pt", padding=True).to(DEVICE)
                    embs = model.get_image_features(**inputs)
                    all_embs.append(embs.cpu())

                combined = torch.cat(all_embs, dim=0)
                mean_emb = combined.mean(dim=0)
                embeddings[pid] = mean_emb

    logger.info(f"Encoded {len(embeddings)} papers, "
                f"{sum(1 for v in embeddings.values() if v is not None)} have figure embeddings")

    # Build aligned tensor
    dim = 512
    aligned = torch.zeros(len(pids_in_order), dim)
    matched = 0
    for i, pid in enumerate(pids_in_order):
        if embeddings.get(pid) is not None:
            aligned[i] = embeddings[pid]
            matched += 1

    logger.info(f"Aligned: {matched}/{len(pids_in_order)} papers matched")

    # L2 normalize (keep zeros for papers without figures)
    norms = aligned.norm(dim=1, keepdim=True)
    norms = norms.clamp(min=1e-8)
    aligned = aligned / norms
    # Restore zeros
    zero_mask = (norms.squeeze() < 1e-7)
    aligned[zero_mask] = 0.0

    zero_count = int(zero_mask.sum())
    logger.info(f"After L2 norm: {zero_count} zero vectors ({zero_count/len(pids_in_order)*100:.1f}%)")

    # Save
    save_data = {
        'ids': pids_in_order,
        'embeddings': aligned,
    }
    save_path = os.path.join(PROCESSED_DIR, 'figure_only_embeddings.pt')
    torch.save(save_data, save_path)
    logger.info(f"Saved figure-only embeddings to {save_path}")
    logger.info(f"Shape: {aligned.shape}")

    # Also regenerate multimodal embeddings with figure-only
    text_data = torch.load(os.path.join(PROCESSED_DIR, 'text_embeddings.pt'), map_location='cpu')
    formula_data = torch.load(os.path.join(PROCESSED_DIR, 'formula_embeddings.pt'), map_location='cpu')

    # Align text and formula to the same paper order
    graph_pids = list(graph['paper_ids'])

    def align_modality(data, graph_pids):
        emb_ids = data['ids']
        emb_tensor = data['embeddings']
        id_to_idx = {pid: i for i, pid in enumerate(emb_ids)}
        d = emb_tensor.size(1)
        result = torch.zeros(len(graph_pids), d)
        for j, pid in enumerate(graph_pids):
            if pid in id_to_idx:
                result[j] = emb_tensor[id_to_idx[pid]]
        return result

    # Use graph paper order for final alignment
    fig_aligned = torch.zeros(len(graph_pids), dim)
    pid_to_idx = {pid: i for i, pid in enumerate(pids_in_order)}
    for j, pid in enumerate(graph_pids):
        if pid in pid_to_idx:
            fig_aligned[j] = aligned[pid_to_idx[pid]]

    text_aligned = align_modality(text_data, graph_pids)
    formula_aligned = align_modality(formula_data, graph_pids)

    # Save multimodal combinations
    for name, parts in [
        ('image', [fig_aligned]),
        ('text+image', [text_aligned, fig_aligned]),
        ('image+formula', [fig_aligned, formula_aligned]),
        ('text+image+formula', [text_aligned, fig_aligned, formula_aligned]),
    ]:
        fused = torch.cat(parts, dim=1)
        path = os.path.join(PROCESSED_DIR, f'multimodal_{name}_embeddings.pt')
        torch.save({'ids': graph_pids, 'embeddings': fused}, path)
        logger.info(f"Saved multimodal_{name}: {fused.shape}")

    logger.info("DONE!")


if __name__ == '__main__':
    main()

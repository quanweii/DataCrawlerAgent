#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
E5 — CLIP 预训练污染 cross-benchmark。
在【外部】图像数据上跑 CLIP-L/14 vs ImageNet-only(ResNet50/ViT-B/16) image-only NC,
测 ΔF1 是否随"该域图像在 LAION 中的常见度"变化(污染假说跨数据集验证)。
  - DTD(Oxford 纹理): LAION 罕见(抽象纹理) → 预期 ΔF1 小
  - amazon(held-out 商品图): LAION 常见(商品照片) → 预期 ΔF1 较大
模型全部本地缓存(CLIP-L/14 / ResNet50 / ViT-B/16), 仅图像数据 DTD 来自 robots.ox.ac.uk。
"""
import os, sys, json, argparse, random
os.environ.setdefault("HF_HUB_OFFLINE", "1")        # 强制用本地缓存(HF 被墙)
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
from pathlib import Path
import numpy as np
import torch, torch.nn as nn
import torch.nn.functional as F
from PIL import Image

ap = argparse.ArgumentParser()
ap.add_argument("--gpu", type=int, default=0)
ap.add_argument("--seeds", type=int, default=5)
ap.add_argument("--epochs", type=int, default=120)
ap.add_argument("--limit_per_class", type=int, default=40)
a = ap.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = str(a.gpu)
DEV = "cuda" if torch.cuda.is_available() else "cpu"
ROOT = Path(__file__).resolve().parents[1]

# ---------- 数据加载 ----------
def load_dtd(per_class):
    from torchvision.datasets import DTD
    ds = DTD(root="/tmp/dtd_data", split="train", download=True)
    # 用全部 split 合并(我们要 NC split, 不是官方 split)
    import torchvision.datasets as DS
    full = DTD(root="/tmp/dtd_data", download=True)  # 默认全部
    by = {}
    for img, lab in full:
        by.setdefault(lab, []).append(img)
    imgs, labs = [], []
    rnd = random.Random(0)
    for lab, lst in by.items():
        rnd.shuffle(lst)
        for im in lst[:per_class]:
            imgs.append(im); labs.append(lab)
    labset = sorted(set(labs)); m = {l:i for i,l in enumerate(labset)}
    return imgs, np.array([m[l] for l in labs]), len(labset)

def load_pets(per_class):
    """Oxford-IIIT Pet (37 猫狗品种)。动物照片 LAION 常见 → 预期 CLIP 有优势。"""
    from torchvision.datasets import OxfordIIITPet
    ds = OxfordIIITPet(root="/tmp/pets_data", download=True)
    by = {}
    for img, lab in ds:
        by.setdefault(lab, []).append(img)
    imgs, labs = [], []
    rnd = random.Random(0)
    for lab, lst in by.items():
        rnd.shuffle(lst)
        for im in lst[:per_class]:
            imgs.append(im); labs.append(lab)
    labset = sorted(set(labs)); m = {l:i for i,l in enumerate(labset)}
    return imgs, np.array([m[l] for l in labs]), len(labset)

def load_amazon(per_class):
    prods = json.load(open(ROOT/"data/amazon_raw/products.json"))
    asin2cat = {p["asin"]: p.get("main_cat","") for p in prods}
    base = ROOT/"data/amazon_products"
    by = {}
    for asin, cat in asin2cat.items():
        if not cat: continue
        d = base/asin/"images"
        if not d.is_dir(): continue
        fs = sorted(f for f in os.listdir(d) if f.lower().endswith((".jpg",".jpeg",".png",".webp")))
        if fs: by.setdefault(cat, []).append(d/fs[0])
    imgs, labs = [], []
    rnd = random.Random(0)
    for cat, lst in by.items():
        if len(lst) < 15: continue   # 只要类够大
        rnd.shuffle(lst)
        for p in lst[:per_class]:
            imgs.append(p); labs.append(cat)
    labset = sorted(set(labs)); m = {l:i for i,l in enumerate(labset)}
    return imgs, np.array([m[l] for l in labs]), len(labset)

# ---------- 编码器 ----------
def clip_encoder():
    from transformers import CLIPModel, CLIPProcessor
    m = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(DEV).eval()
    proc = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
    def enc(paths_or_imgs, batch=32):
        feats = []
        for s in range(0, len(paths_or_imgs), batch):
            chunk = paths_or_imgs[s:s+batch]
            imgs = [(Image.open(p).convert("RGB") if isinstance(p, Path) else p) for p in chunk]
            inp = proc(images=imgs, return_tensors="pt").to(DEV)
            with torch.no_grad():
                f = m.get_image_features(**inp)   # 768d projected
            feats.append(f.cpu())
        return torch.cat(feats).numpy()
    return enc, 768

def in1k_encoder(name):
    import torchvision.models as M, torchvision.transforms as T
    if name == "resnet50":
        from torchvision.models import ResNet50_Weights
        m = M.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1); m.fc = nn.Identity(); dim=2048
        tfm = T.Compose([T.Resize(256), T.CenterCrop(224), T.ToTensor(),
                         T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])
    else:
        from torchvision.models import ViT_B_16_Weights
        w = ViT_B_16_Weights.IMAGENET1K_V1; m = M.vit_b_16(weights=w); m.heads = nn.Identity(); dim=768
        tfm = w.transforms()
    m = m.to(DEV).eval()
    def enc(paths_or_imgs, batch=64):
        feats = []
        for s in range(0, len(paths_or_imgs), batch):
            chunk = [tfm(Image.open(p).convert("RGB") if isinstance(p,Path) else p) for p in paths_or_imgs[s:s+batch]]
            with torch.no_grad():
                feats.append(m(torch.stack(chunk).to(DEV)).cpu())
        return torch.cat(feats).numpy()
    return enc, dim

# ---------- MLP NC (no graph) ----------
def stratified(y, ratios=(0.6,0.2,0.2), seed=1):
    rnd = np.random.RandomState(seed); n=len(y); tr=va=te=np.zeros(n,bool)
    for c in np.unique(y):
        idx = np.where(y==c)[0]; rnd.shuffle(idx)
        i6=int(len(idx)*0.6); i8=int(len(idx)*0.8)
        tr[idx[:i6]]=True; va[idx[i6:i8]]=True; te[idx[i8:]]=True
    return tr, va, te

def mlp_nc(X, y, ncls, seeds, epochs):
    from sklearn.metrics import f1_score
    Xt = torch.tensor(X, dtype=torch.float32)
    yt = torch.tensor(y, dtype=torch.long)
    f1s=[]
    for s in range(seeds):
        tr,va,te = stratified(y, seed=s)
        din = X.shape[1]
        net = nn.Sequential(nn.Linear(din,256), nn.ReLU(), nn.Dropout(0.5), nn.Linear(256,ncls)).to(DEV)
        opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=5e-4)
        best=-1; best_f1=0; patience=0
        for ep in range(epochs):
            net.train(); opt.zero_grad()
            loss = F.cross_entropy(net(Xt[tr].to(DEV)), yt[tr].to(DEV))
            loss.backward(); opt.step()
            if ep%5==0 or ep==epochs-1:
                net.eval()
                with torch.no_grad():
                    pred = net(Xt[te].to(DEV)).argmax(1).cpu().numpy()
                f = f1_score(y[te], pred, average="macro")
                if f>best_f1: best_f1=f; patience=0
                else:
                    patience+=1
                    if patience>8: break
        f1s.append(best_f1)
    return float(np.mean(f1s)), float(np.std(f1s))

def run_dataset(name, loader, per_class):
    print(f"\n=== {name} ===", flush=True)
    imgs, y, ncls = loader(per_class)
    print(f"  {len(imgs)} imgs, {ncls} classes", flush=True)
    encoders = {"clip-vitl14": clip_encoder(), "resnet50_in1k": in1k_encoder("resnet50"),
                "vitb16_in1k": in1k_encoder("vitb16")}
    res = {}
    for ename, (enc, dim) in encoders.items():
        X = enc(imgs); X = X/np.maximum(np.linalg.norm(X,axis=1,keepdims=True),1e-8)
        f1m,f1s = mlp_nc(X, y, ncls, a.seeds, a.epochs)
        res[ename] = {"f1": round(f1m,3), "std": round(f1s,3), "dim": int(dim)}
        print(f"  {ename:14s} F1={f1m:.3f}±{f1s:.3f}", flush=True)
    clip = res["clip-vitl14"]["f1"]
    for k in ("resnet50_in1k","vitb16_in1k"):
        res[k]["delta_clip"] = round(clip - res[k]["f1"], 3)
    print(f"  Δ(CLIP−resnet50)={res['resnet50_in1k']['delta_clip']:+.3f}  Δ(CLIP−vitb16)={res['vitb16_in1k']['delta_clip']:+.3f}", flush=True)
    return res

def main():
    out = {}
    out["DTD_textures_LAIONrare"] = run_dataset("DTD (Oxford textures, LAION-rare)", load_dtd, a.limit_per_class)
    out["Pets_animals_LAIONcommon"] = run_dataset("Oxford Pets (cat/dog breeds, LAION-common)", load_pets, a.limit_per_class)
    out["amazon_products"] = run_dataset("amazon (held-out product photos)", load_amazon, a.limit_per_class)
    json.dump(out, open(ROOT/"benchmark/results/e5_cross_clip_contamination.json","w"), indent=2)
    print("\nsaved -> benchmark/results/e5_cross_clip_contamination.json")

if __name__ == "__main__":
    main()

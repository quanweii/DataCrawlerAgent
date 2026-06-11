"""
图像编码器 - 使用 CLIP-ViT-Large-Patch14 编码论文图片
统一输出768维，与文本/公式编码器维度一致
每篇论文的多张图片编码后取平均池化
"""
import os
import torch
import numpy as np
from typing import List, Optional, Dict
from loguru import logger
from tqdm import tqdm
from PIL import Image


class ImageEncoder:
    """论文图像编码器，输出768维CLIP图像特征"""

    def __init__(
        self,
        model_name: str = "openai/clip-vit-large-patch14",
        device: str = "cuda",
        batch_size: int = 16,
    ):
        self.device = device
        self.batch_size = batch_size

        from transformers import CLIPModel, CLIPProcessor
        self.model = CLIPModel.from_pretrained(model_name).to(device)
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.model.eval()
        self.embedding_dim = self.model.config.projection_dim  # 768
        logger.info(f"ImageEncoder loaded: {model_name}, dim={self.embedding_dim}")

    @torch.no_grad()
    def encode_images(self, image_paths: List[str]) -> np.ndarray:
        """编码多张图片，返回平均池化的embedding"""
        if not image_paths:
            return np.zeros(self.embedding_dim, dtype=np.float32)

        valid_images = []
        for path in image_paths:
            try:
                img = Image.open(path).convert("RGB")
                valid_images.append(img)
            except Exception:
                continue

        if not valid_images:
            return np.zeros(self.embedding_dim, dtype=np.float32)

        embeddings = []
        for i in range(0, len(valid_images), self.batch_size):
            batch = valid_images[i:i + self.batch_size]
            inputs = self.processor(images=batch, return_tensors="pt", padding=True)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            outputs = self.model.get_image_features(**inputs)
            embeddings.append(outputs.cpu().numpy())

        all_embs = np.concatenate(embeddings, axis=0)
        mean_emb = np.mean(all_embs, axis=0).astype(np.float32)
        norm = np.linalg.norm(mean_emb)
        if norm > 1e-8:
            mean_emb = mean_emb / norm
        return mean_emb

    def encode_batch(
        self,
        paper_image_dirs: Dict[str, str],
        output_dir: str,
    ) -> dict:
        """批量编码论文图片"""
        os.makedirs(output_dir, exist_ok=True)

        all_ids = []
        all_embeddings = []

        for paper_id, img_dir in tqdm(paper_image_dirs.items(), desc="Encoding images"):
            image_paths = self._list_images(img_dir)
            emb = self.encode_images(image_paths)
            all_ids.append(paper_id)
            all_embeddings.append(emb)

        embeddings = torch.tensor(np.stack(all_embeddings))
        save_path = os.path.join(output_dir, "image_embeddings.pt")
        torch.save({"ids": all_ids, "embeddings": embeddings}, save_path)

        logger.info(f"图像编码完成: {len(all_ids)} 篇, dim={embeddings.size(1)}")
        return {"path": save_path, "ids": all_ids, "dim": self.embedding_dim}

    def _list_images(self, img_dir: str) -> List[str]:
        if not os.path.isdir(img_dir):
            return []
        exts = {".png", ".jpg", ".jpeg", ".bmp", ".tiff"}
        return sorted([
            os.path.join(img_dir, f)
            for f in os.listdir(img_dir)
            if os.path.splitext(f)[1].lower() in exts
        ])

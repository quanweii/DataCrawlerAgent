"""
公式编码器 - 从markdown中提取LaTeX公式并编码
使用 CLIP-ViT-Large-Patch14 文本分支编码，统一输出768维
"""
import os
import re
import torch
import numpy as np
from typing import List, Dict, Optional
from loguru import logger
from tqdm import tqdm


class FormulaEncoder:
    """论文公式编码器，输出768维CLIP文本特征"""

    def __init__(
        self,
        model_name: str = "openai/clip-vit-large-patch14",
        device: str = "cuda",
        batch_size: int = 64,
        max_formulas: int = 200,
    ):
        self.device = device
        self.batch_size = batch_size
        self.max_formulas = max_formulas

        from transformers import CLIPModel, CLIPTokenizer
        self.model = CLIPModel.from_pretrained(model_name).to(device)
        self.tokenizer = CLIPTokenizer.from_pretrained(model_name)
        self.model.eval()
        self.embedding_dim = self.model.config.projection_dim  # 768
        self.max_tokens = self.model.config.text_config.max_position_embeddings  # 77
        logger.info(f"FormulaEncoder loaded: {model_name}, dim={self.embedding_dim}")

    def extract_formulas(self, text: str) -> List[str]:
        """从markdown文本中提取LaTeX公式"""
        block_formulas = re.findall(r'\$\$(.+?)\$\$', text, re.DOTALL)
        inline_formulas = re.findall(r'(?<!\$)\$(?!\$)(.+?)(?<!\$)\$(?!\$)', text)
        formulas = block_formulas + inline_formulas
        formulas = [f.strip() for f in formulas if len(f.strip()) > 5]
        return formulas[:self.max_formulas]

    @torch.no_grad()
    def encode_formulas(self, formulas: List[str]) -> np.ndarray:
        """编码一组公式，返回平均池化的embedding"""
        if not formulas:
            return np.zeros(self.embedding_dim, dtype=np.float32)

        embeddings = []
        for i in range(0, len(formulas), self.batch_size):
            batch = formulas[i:i + self.batch_size]
            inputs = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=self.max_tokens,
                return_tensors="pt",
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            outputs = self.model.get_text_features(**inputs)
            outputs = torch.nn.functional.normalize(outputs, p=2, dim=1)
            embeddings.append(outputs.cpu().numpy())

        all_embs = np.concatenate(embeddings, axis=0)
        return np.mean(all_embs, axis=0).astype(np.float32)

    def encode_batch(
        self,
        md_paths: List[str],
        ids: List[str],
        output_dir: str,
    ) -> dict:
        """批量编码论文公式"""
        os.makedirs(output_dir, exist_ok=True)

        all_ids = []
        all_embeddings = []
        total_formulas = 0

        for md_path, paper_id in tqdm(zip(md_paths, ids), total=len(md_paths), desc="Encoding formulas"):
            text = self._read_md(md_path)
            formulas = self.extract_formulas(text)
            emb = self.encode_formulas(formulas)
            all_ids.append(paper_id)
            all_embeddings.append(emb)
            total_formulas += len(formulas)

        embeddings = torch.tensor(np.stack(all_embeddings))
        save_path = os.path.join(output_dir, "formula_embeddings.pt")
        torch.save({"ids": all_ids, "embeddings": embeddings}, save_path)

        logger.info(f"公式编码完成: {len(all_ids)} 篇, 共 {total_formulas} 条公式, "
                     f"dim={embeddings.size(1)}")
        return {"path": save_path, "ids": all_ids, "dim": self.embedding_dim}

    def _read_md(self, md_path: str) -> str:
        try:
            with open(md_path, "r", encoding="utf-8") as f:
                return f.read()
        except Exception:
            return ""

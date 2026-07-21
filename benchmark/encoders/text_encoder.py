"""
文本编码器 - 使用 CLIP-ViT-Large-Patch14 编码论文文本
统一输出768维，与图像/公式编码器维度一致
支持长文档分段编码 + 池化

2026-07-15: 增加 strip_patterns 参数，用于在编码前清洗文本中的
label 承载字段（如 "Style: X" / "Genres: X"），消除特征级 label 泄露。
"""
import os
import re
import torch
import numpy as np
from typing import List, Optional
from loguru import logger
from tqdm import tqdm


class TextEncoder:
    """论文文本编码器，输出768维CLIP文本特征"""

    def __init__(
        self,
        model_name: str = "openai/clip-vit-large-patch14",
        device: str = "cuda",
        chunk_size: int = 200,
        batch_size: int = 32,
        strip_patterns: Optional[List[str]] = None,
    ):
        """
        strip_patterns: 可选的正则列表。每条在 _read_md 之后以 re.MULTILINE|re.IGNORECASE
        匹配整行，命中则删除该行。用于清洗 label-bearing 元数据字段。
        例如 wikiart: [r'^Style\s*:.*$'] ; steam: [r'^Genres?\s*:.*$']
        """
        self.device = device
        self.chunk_size = chunk_size
        self.batch_size = batch_size
        self._strip_res = None
        if strip_patterns:
            self._strip_res = [re.compile(p, re.MULTILINE | re.IGNORECASE) for p in strip_patterns]

        from transformers import CLIPModel, CLIPTokenizer
        self.model = CLIPModel.from_pretrained(model_name).to(device)
        self.tokenizer = CLIPTokenizer.from_pretrained(model_name)
        self.model.eval()
        self.embedding_dim = self.model.config.projection_dim  # 768
        self.max_tokens = self.model.config.text_config.max_position_embeddings  # 77
        strip_note = f", strip_patterns={strip_patterns}" if strip_patterns else ""
        logger.info(f"TextEncoder loaded: {model_name}, dim={self.embedding_dim}{strip_note}")

    @torch.no_grad()
    def encode_single(self, text: str) -> np.ndarray:
        """编码单篇论文的文本（支持长文档分段）"""
        if not text or len(text.strip()) < 10:
            return np.zeros(self.embedding_dim, dtype=np.float32)

        chunks = self._chunk_text(text)
        if not chunks:
            return np.zeros(self.embedding_dim, dtype=np.float32)

        embeddings = []
        for i in range(0, len(chunks), self.batch_size):
            batch_texts = chunks[i:i + self.batch_size]
            inputs = self.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=self.max_tokens,
                return_tensors="pt",
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            outputs = self.model.get_text_features(**inputs)
            # L2 normalize each chunk
            outputs = torch.nn.functional.normalize(outputs, p=2, dim=1)
            embeddings.append(outputs.cpu().numpy())

        all_embs = np.concatenate(embeddings, axis=0)
        return np.mean(all_embs, axis=0).astype(np.float32)

    def encode_batch(
        self,
        md_paths: List[str],
        output_dir: str,
        ids: Optional[List[str]] = None,
    ) -> dict:
        """
        批量编码论文文本，保存为 .pt 文件
        """
        os.makedirs(output_dir, exist_ok=True)

        all_embeddings = []
        all_ids = ids or [os.path.splitext(os.path.basename(p))[0] for p in md_paths]

        for i, (md_path, paper_id) in enumerate(tqdm(
            zip(md_paths, all_ids), total=len(md_paths), desc="Encoding text"
        )):
            text = self._read_md(md_path)
            emb = self.encode_single(text)
            all_embeddings.append(emb)

        embeddings = torch.tensor(np.stack(all_embeddings))
        save_path = os.path.join(output_dir, "text_embeddings.pt")
        torch.save({"ids": all_ids, "embeddings": embeddings}, save_path)

        logger.info(f"文本编码完成: {len(all_ids)} 篇, dim={embeddings.size(1)}, "
                     f"保存到 {save_path}")

        return {"path": save_path, "ids": all_ids, "dim": self.embedding_dim}

    def _chunk_text(self, text: str) -> List[str]:
        """将长文本按段落分段"""
        paragraphs = text.split("\n\n")
        chunks = []
        current_chunk = ""

        for para in paragraphs:
            para = para.strip()
            if not para:
                continue
            if len(current_chunk) + len(para) < self.chunk_size:
                current_chunk += " " + para
            else:
                if current_chunk.strip():
                    chunks.append(current_chunk.strip())
                current_chunk = para

        if current_chunk.strip():
            chunks.append(current_chunk.strip())

        return chunks

    def _read_md(self, md_path: str) -> str:
        try:
            with open(md_path, "r", encoding="utf-8") as f:
                text = f.read()
        except Exception:
            return ""
        if self._strip_res:
            for pat in self._strip_res:
                text = pat.sub("", text)
        return text

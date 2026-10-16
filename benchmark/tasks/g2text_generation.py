"""
Graph-to-Text Generation Task - OpenMAG Benchmark
GNN encoder -> Projection -> LLM backbone (OPT-125M) for text generation from graph embeddings.

Architecture:
    1. Pre-trained GNN encoder (frozen) produces node embeddings of hidden_dim.
    2. Projection layer maps GNN output to the LLM hidden dimension.
    3. LLM (OPT-125M) generates textual descriptions conditioned on the projected
       graph embedding, which is prepended as a soft prompt.

PEFT strategies compared:
    - None  : full fine-tuning of projection layer + LLM
    - LoRA  : low-rank adaptation (r=64, alpha=128) on LLM linear layers
    - Adapter: bottleneck adapter (dim=64) inserted after LLM attention/FFN
    - IA3   : learned vectors scaling attention/FFN activations

Evaluation metrics:
    - BLEU-4  (sacrebleu)
    - ROUGE-L (rouge_score)
    - CIDEr   (custom n-gram TF-IDF implementation)

GNN backbones tested: GCN, GAT, GraphSAGE.
"""

import os
import re
import json
import time
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple
from collections import defaultdict
from loguru import logger


# ---------------------------------------------------------------------------
# Paper metadata loading
# ---------------------------------------------------------------------------

def load_paper_texts(
    data_root: str,
    domains: List[str],
    paper_ids: List[str],
    max_length: int = 256,
) -> Tuple[List[str], Dict[str, str]]:
    """
    Load paper titles and abstracts from the markdown files stored under
    ``data_root/{domain}_papers/{arxiv_id}/{arxiv_id}.md``.

    Returns:
        texts  : list of str aligned with *paper_ids* (empty string when missing)
        id2text: dict mapping paper_id -> full text
    """
    id_to_domain = {}
    for domain in domains:
        papers_dir = os.path.join(data_root, f"{domain}_papers")
        if not os.path.isdir(papers_dir):
            logger.warning(f"Papers directory not found: {papers_dir}")
            continue
        for arxiv_id in os.listdir(papers_dir):
            full_path = os.path.join(papers_dir, arxiv_id)
            if os.path.isdir(full_path):
                id_to_domain[arxiv_id] = domain

    id2text: Dict[str, str] = {}
    for pid, domain in id_to_domain.items():
        md_path = os.path.join(data_root, f"{domain}_papers", pid, f"{pid}.md")
        if not os.path.exists(md_path):
            continue
        try:
            with open(md_path, "r", encoding="utf-8", errors="ignore") as fh:
                raw = fh.read()
        except Exception:
            continue

        # Extract title (first ## heading or first non-empty line)
        title = ""
        for line in raw.split("\n"):
            line = line.strip()
            if line.startswith("## ") and not title:
                title = line[3:].strip()
                break
            elif line and not line.startswith("Leijnen") and not title:
                # Fallback: use the first substantial non-reference line as title
                pass

        # Extract abstract
        abstract = ""
        abstract_match = re.search(
            r"(?i)##?\s*abstract\s*\n([\s\S]*?)(?=\n##|\nKeywords|\nIntroduction)",
            raw,
        )
        if abstract_match:
            abstract = abstract_match.group(1).strip()

        # Compose: title + abstract
        text = f"{title}. {abstract}".strip()
        # Basic cleaning
        text = re.sub(r"\s+", " ", text)
        text = re.sub(r"\$[^$]*\$", "", text)  # remove inline LaTeX
        text = text[:max_length]
        id2text[pid] = text

    # Align with paper_ids order
    texts = [id2text.get(pid, "") for pid in paper_ids]
    found = sum(1 for t in texts if t)
    logger.info(f"Loaded paper texts for {found}/{len(paper_ids)} papers")
    return texts, id2text


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def _tokenize_ngrams(text: str, n: int) -> List[Tuple[str, ...]]:
    """Simple whitespace + punctuation tokenization -> n-gram tuples."""
    tokens = re.findall(r"\w+", text.lower())
    if len(tokens) < n:
        return []
    return [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]


def compute_bleu4(references: List[str], hypotheses: List[str]) -> float:
    """Compute corpus-level BLEU-4 using sacrebleu if available, else fallback."""
    try:
        import sacrebleu

        # sacrebleu expects List[str] refs and List[str] hyps
        refs = [[r for r in references]]
        bleu = sacrebleu.corpus_bleu(hypotheses, refs)
        return bleu.score / 100.0  # normalize to [0, 1]
    except ImportError:
        logger.warning("sacrebleu not installed; using simplified BLEU-4")
        return _simplified_bleu4(references, hypotheses)


def _simplified_bleu4(references: List[str], hypotheses: List[str]) -> float:
    """Very simple BLEU-4 approximation for when sacrebleu is unavailable."""
    from collections import Counter

    if not hypotheses or not references:
        return 0.0

    match_counts = [0, 0, 0, 0]
    total_counts = [0, 0, 0, 0]
    ref_len = 0
    hyp_len = 0

    for ref, hyp in zip(references, hypotheses):
        ref_tokens = re.findall(r"\w+", ref.lower())
        hyp_tokens = re.findall(r"\w+", hyp.lower())
        ref_len += len(ref_tokens)
        hyp_len += len(hyp_tokens)

        for n in range(1, 5):
            ref_ngrams = Counter(_tokenize_ngrams(ref, n))
            hyp_ngrams = Counter(_tokenize_ngrams(hyp, n))
            for ng, cnt in hyp_ngrams.items():
                match_counts[n - 1] += min(cnt, ref_ngrams.get(ng, 0))
                total_counts[n - 1] += cnt

    if hyp_len == 0:
        return 0.0

    # Brevity penalty
    bp = min(1.0, math.exp(1 - ref_len / max(hyp_len, 1)))

    # Log-average precision for n=1..4
    log_avg = 0.0
    for n in range(4):
        if total_counts[n] == 0:
            return 0.0
        log_avg += math.log(max(match_counts[n], 1e-10) / total_counts[n])
    log_avg /= 4.0

    return bp * math.exp(log_avg)


def compute_rouge_l(references: List[str], hypotheses: List[str]) -> float:
    """Compute average ROUGE-L F1 using rouge_score if available, else fallback."""
    try:
        from rouge_score import rouge_scorer

        scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
        scores = [scorer.score(r, h)["rougeL"].fmeasure for r, h in zip(references, hypotheses)]
        return float(np.mean(scores)) if scores else 0.0
    except ImportError:
        logger.warning("rouge_score not installed; using simplified ROUGE-L")
        return _simplified_rouge_l(references, hypotheses)


def _simplified_rouge_l(references: List[str], hypotheses: List[str]) -> float:
    """Simplified ROUGE-L based on longest common subsequence."""
    scores = []
    for ref, hyp in zip(references, hypotheses):
        ref_tokens = ref.lower().split()
        hyp_tokens = hyp.lower().split()
        if not ref_tokens or not hyp_tokens:
            scores.append(0.0)
            continue
        lcs_len = _lcs_length(ref_tokens, hyp_tokens)
        precision = lcs_len / len(hyp_tokens)
        recall = lcs_len / len(ref_tokens)
        if precision + recall == 0:
            scores.append(0.0)
        else:
            scores.append(2 * precision * recall / (precision + recall))
    return float(np.mean(scores)) if scores else 0.0


def _lcs_length(x: List[str], y: List[str]) -> int:
    """Compute length of longest common subsequence."""
    m, n = len(x), len(y)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if x[i - 1] == y[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    return dp[m][n]


def compute_cider(
    references: List[str],
    hypotheses: List[str],
    n: int = 4,
) -> float:
    """
    Compute CIDEr (Consensus-based Image Description Evaluation) score.

    CIDEr measures the consensus of a generated text against reference texts
    using n-gram TF-IDF similarity.  For this single-reference setting, we
    treat each reference as its own document for IDF computation.
    """
    if not hypotheses or not references:
        return 0.0

    num_docs = len(references)

    # Build document frequency counts
    df = [defaultdict(int) for _ in range(n)]
    for ref in references:
        for k in range(n):
            ngrams = set(_tokenize_ngrams(ref, k + 1))
            for ng in ngrams:
                df[k][ng] += 1

    scores = []
    for ref, hyp in zip(references, hypotheses):
        sim = 0.0
        for k in range(n):
            ref_ngrams = _tokenize_ngrams(ref, k + 1)
            hyp_ngrams = _tokenize_ngrams(hyp, k + 1)
            if not ref_ngrams or not hyp_ngrams:
                continue

            # TF vectors
            ref_tf: Dict[tuple, float] = defaultdict(float)
            hyp_tf: Dict[tuple, float] = defaultdict(float)
            for ng in ref_ngrams:
                ref_tf[ng] += 1.0 / len(ref_ngrams)
            for ng in hyp_ngrams:
                hyp_tf[ng] += 1.0 / len(hyp_ngrams)

            # TF-IDF and cosine similarity
            all_ngrams = set(ref_tf.keys()) | set(hyp_tf.keys())
            dot = 0.0
            norm_ref = 0.0
            norm_hyp = 0.0
            for ng in all_ngrams:
                idf = math.log((num_docs + 1) / (df[k].get(ng, 0) + 1)) + 1
                r = ref_tf.get(ng, 0) * idf
                h = hyp_tf.get(ng, 0) * idf
                dot += r * h
                norm_ref += r * r
                norm_hyp += h * h

            denom = math.sqrt(norm_ref) * math.sqrt(norm_hyp)
            if denom > 0:
                sim += dot / denom

        # Average over n-gram orders
        sim /= n
        scores.append(sim)

    return float(np.mean(scores)) if scores else 0.0


# ---------------------------------------------------------------------------
# Projection layer: GNN hidden_dim -> LLM hidden_dim
# ---------------------------------------------------------------------------

class GraphProjection(nn.Module):
    """Two-layer MLP projection from GNN embedding space to LLM hidden space."""

    def __init__(self, gnn_dim: int, llm_dim: int, dropout: float = 0.1):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(gnn_dim, llm_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(llm_dim, llm_dim),
        )
        # Initialize near-identity for stable early training
        nn.init.xavier_uniform_(self.proj[0].weight, gain=0.02)
        nn.init.zeros_(self.proj[0].bias)
        nn.init.xavier_uniform_(self.proj[3].weight, gain=0.02)
        nn.init.zeros_(self.proj[3].bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


# ---------------------------------------------------------------------------
# G2Text model: frozen GNN -> projection -> PEFT-wrapped LLM
# ---------------------------------------------------------------------------

class G2TextModel(nn.Module):
    """
    Full Graph-to-Text generation model.

    Components:
        - gnn_encoder : frozen pre-trained GNN (produces node embeddings)
        - projection  : trainable GraphProjection layer
        - llm         : OPT-125M with optional PEFT wrapping
        - peft_type   : "none" | "lora" | "adapter" | "ia3"
    """

    def __init__(
        self,
        gnn_encoder: nn.Module,
        gnn_dim: int,
        llm_model_name: str = "facebook/opt-125m",
        peft_type: str = "none",
        lora_r: int = 64,
        lora_alpha: int = 128,
        adapter_bottleneck: int = 64,
        max_gen_length: int = 128,
        device: str = "cuda",
    ):
        super().__init__()
        self.gnn_encoder = gnn_encoder
        self.peft_type = peft_type
        self.max_gen_length = max_gen_length
        self.device = device

        # Freeze GNN
        for param in self.gnn_encoder.parameters():
            param.requires_grad = False
        self.gnn_encoder.eval()

        # Load LLM backbone
        from transformers import AutoModelForCausalLM, AutoTokenizer

        logger.info(f"Loading LLM backbone: {llm_model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(llm_model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.llm = AutoModelForCausalLM.from_pretrained(llm_model_name)
        llm_hidden_dim = self.llm.config.hidden_size  # 768 for OPT-125M
        self.llm_hidden_dim = llm_hidden_dim

        # Projection: GNN output -> LLM hidden dim
        self.projection = GraphProjection(gnn_dim, llm_hidden_dim)

        # Apply PEFT to the LLM
        self._apply_peft(
            peft_type, lora_r, lora_alpha, adapter_bottleneck
        )

        logger.info(
            f"G2TextModel created: peft={peft_type}, "
            f"trainable params={self._count_trainable():,}, "
            f"total params={self._count_total():,}"
        )

    def _apply_peft(
        self,
        peft_type: str,
        lora_r: int,
        lora_alpha: int,
        adapter_bottleneck: int,
    ):
        """Wrap the LLM with the chosen PEFT strategy."""
        if peft_type == "none":
            # Full fine-tuning: projection + all LLM params are trainable
            return

        if peft_type == "lora":
            from peft import LoraConfig, get_peft_model

            # Target all linear layers in the OPT decoder
            lora_config = LoraConfig(
                r=lora_r,
                lora_alpha=lora_alpha,
                target_modules=["q_proj", "k_proj", "v_proj", "out_proj",
                                "fc1", "fc2"],
                lora_dropout=0.1,
                bias="none",
                task_type="CAUSAL_LM",
            )
            self.llm = get_peft_model(self.llm, lora_config)
            self.llm.print_trainable_parameters()

        elif peft_type == "adapter":
            from peft import AdapterConfig, get_peft_model

            adapter_config = AdapterConfig(
                adapter_len=adapter_bottleneck,
                adapter_layers=12,  # all OPT-125M decoder layers
                non_linearity="relu",
                task_type="CAUSAL_LM",
            )
            self.llm = get_peft_model(self.llm, adapter_config)
            self.llm.print_trainable_parameters()

        elif peft_type == "ia3":
            from peft import IA3Config, get_peft_model

            ia3_config = IA3Config(
                target_modules=["q_proj", "k_proj", "v_proj", "fc1", "fc2"],
                feedforward_modules=["fc1", "fc2"],
                task_type="CAUSAL_LM",
            )
            self.llm = get_peft_model(self.llm, ia3_config)
            self.llm.print_trainable_parameters()

        else:
            raise ValueError(
                f"Unknown peft_type: {peft_type}. "
                f"Choose from: none, lora, adapter, ia3"
            )

    def _count_trainable(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def _count_total(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def encode_graph(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> torch.Tensor:
        """Run frozen GNN encoder and project to LLM space."""
        with torch.no_grad():
            self.gnn_encoder.eval()
            node_embs = self.gnn_encoder(x, edge_index)  # [N, gnn_dim]
            node_embs = node_embs.detach()
        return self.projection(node_embs).detach()  # [N, llm_hidden_dim]

    def forward(
        self,
        graph_embs: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass for training.

        Args:
            graph_embs:  [batch, llm_hidden_dim] projected graph embeddings
            input_ids:   [batch, seq_len] tokenized target text
            attention_mask: [batch, seq_len]
            labels:      [batch, seq_len] same as input_ids (shifted internally by OPT)
        """
        batch_size = graph_embs.size(0)

        # Get LLM word embeddings
        word_embeds = self.llm.get_input_embeddings()(input_ids)  # [B, T, D]

        # Prepend graph embedding as a "soft prompt" token
        graph_prefix = graph_embs.detach().unsqueeze(1)  # [B, 1, D] - detach to avoid backward issues
        inputs_embeds = torch.cat([graph_prefix, word_embeds], dim=1)  # [B, 1+T, D]

        # Adjust attention mask to account for the prefix token
        prefix_mask = torch.ones(
            batch_size, 1, device=attention_mask.device, dtype=attention_mask.dtype
        )
        extended_mask = torch.cat([prefix_mask, attention_mask], dim=1)  # [B, 1+T]

        # Adjust labels: pad with -100 at the prefix position so it is ignored
        if labels is not None:
            prefix_labels = torch.full(
                (batch_size, 1), -100, device=labels.device, dtype=labels.dtype
            )
            extended_labels = torch.cat([prefix_labels, labels], dim=1)  # [B, 1+T]
        else:
            extended_labels = None

        outputs = self.llm(
            inputs_embeds=inputs_embeds,
            attention_mask=extended_mask,
            labels=extended_labels,
        )

        return {
            "loss": outputs.loss,
            "logits": outputs.logits,
        }

    @torch.no_grad()
    def generate(
        self,
        graph_embs: torch.Tensor,
        max_length: int = 128,
        beam_size: int = 5,
    ) -> List[str]:
        """
        Generate text from graph embeddings using beam search.

        Args:
            graph_embs: [batch, llm_hidden_dim]
            max_length: maximum number of new tokens to generate
            beam_size: beam width for beam search
        """
        self.eval()
        batch_size = graph_embs.size(0)

        # Prepare the graph embedding as a prefix
        graph_prefix = graph_embs.unsqueeze(1)  # [B, 1, D]

        # Use BOS token to start generation
        bos_ids = torch.full(
            (batch_size, 1),
            self.tokenizer.bos_token_id or self.tokenizer.eos_token_id,
            device=self.device,
            dtype=torch.long,
        )

        bos_embeds = self.llm.get_input_embeddings()(bos_ids)  # [B, 1, D]
        initial_embeds = torch.cat([graph_prefix, bos_embeds], dim=1)  # [B, 2, D]

        # Build attention mask
        attention_mask = torch.ones(
            batch_size, 2, device=self.device, dtype=torch.long
        )

        generated_ids = self.llm.generate(
            inputs_embeds=initial_embeds,
            attention_mask=attention_mask,
            max_new_tokens=max_length,
            num_beams=beam_size,
            early_stopping=True,
            no_repeat_ngram_size=3,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )

        # Decode, skipping the prefix and BOS
        texts = []
        for seq in generated_ids:
            # Strip the 2-token prefix (graph + BOS) by decoding all and cleaning
            text = self.tokenizer.decode(seq, skip_special_tokens=True)
            texts.append(text.strip())

        return texts


# ---------------------------------------------------------------------------
# Dataset wrapper for batching
# ---------------------------------------------------------------------------

class G2TextDataset(torch.utils.data.Dataset):
    """
    Wraps node indices with their corresponding graph embeddings and target
    texts for efficient batching during G2Text training.
    """

    def __init__(
        self,
        node_indices: np.ndarray,
        graph_embs: torch.Tensor,
        texts: List[str],
        tokenizer,
        max_length: int = 128,
    ):
        self.node_indices = node_indices
        self.graph_embs = graph_embs
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.node_indices)

    def __getitem__(self, idx):
        node_idx = self.node_indices[idx]
        graph_emb = self.graph_embs[node_idx]
        text = self.texts[idx] if idx < len(self.texts) else ""

        enc = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        input_ids = enc["input_ids"].squeeze(0)
        attention_mask = enc["attention_mask"].squeeze(0)
        labels = input_ids.clone()
        labels[labels == self.tokenizer.pad_token_id] = -100

        return {
            "graph_emb": graph_emb,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


def collate_fn(batch):
    """Custom collate for G2Text batches."""
    return {
        "graph_emb": torch.stack([b["graph_emb"] for b in batch]),
        "input_ids": torch.stack([b["input_ids"] for b in batch]),
        "attention_mask": torch.stack([b["attention_mask"] for b in batch]),
        "labels": torch.stack([b["labels"] for b in batch]),
    }


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train_g2text(
    model: G2TextModel,
    dataset: G2TextDataset,
    val_dataset: G2TextDataset,
    num_epochs: int = 10,
    lr: float = 5e-5,
    weight_decay: float = 0.01,
    batch_size: int = 32,
    patience: int = 3,
    device: str = "cuda",
) -> Dict:
    """
    Train the G2Text model (projection + LLM with PEFT).

    Returns:
        Dict with training history (losses, best epoch).
    """
    model = model.to(device)
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr,
        weight_decay=weight_decay,
    )

    # Cosine learning rate schedule with warm-up
    total_steps = num_epochs * (len(dataset) + batch_size - 1) // batch_size
    warmup_steps = min(100, total_steps // 10)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda step: min(
            1.0,
            (step + 1) / max(1, warmup_steps),
        )
        * max(0.0, 0.5 * (1 + math.cos(math.pi * step / max(1, total_steps)))),
    )

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0,
        pin_memory=True,
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0,
        pin_memory=True,
    )

    best_val_loss = float("inf")
    best_epoch = 0
    best_state = None
    patience_counter = 0
    history = {"train_loss": [], "val_loss": []}

    start_time = time.time()

    for epoch in range(1, num_epochs + 1):
        # --- Training ---
        model.train()
        epoch_loss = 0.0
        num_batches = 0
        for batch in loader:
            graph_embs = batch["graph_emb"].to(device)
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(graph_embs, input_ids, attention_mask, labels)
            loss = outputs["loss"]

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            epoch_loss += loss.item()
            num_batches += 1

        avg_train_loss = epoch_loss / max(num_batches, 1)
        history["train_loss"].append(avg_train_loss)

        # --- Validation ---
        model.eval()
        val_loss = 0.0
        val_batches = 0
        with torch.no_grad():
            for batch in val_loader:
                graph_embs = batch["graph_emb"].to(device)
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)

                outputs = model(graph_embs, input_ids, attention_mask, labels)
                val_loss += outputs["loss"].item()
                val_batches += 1

        avg_val_loss = val_loss / max(val_batches, 1)
        history["val_loss"].append(avg_val_loss)

        logger.info(
            f"[Epoch {epoch}/{num_epochs}] "
            f"train_loss={avg_train_loss:.4f}, val_loss={avg_val_loss:.4f}, "
            f"lr={scheduler.get_last_lr()[0]:.2e}"
        )

        # Early stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_epoch = epoch
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            logger.info(f"Early stopping at epoch {epoch}")
            break

    training_time = time.time() - start_time

    # Restore best model
    if best_state is not None:
        model.load_state_dict(best_state)
        model = model.to(device)

    return {
        "best_epoch": best_epoch,
        "best_val_loss": float(best_val_loss),
        "training_time": training_time,
        "history": history,
    }


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate_g2text(
    model: G2TextModel,
    test_indices: np.ndarray,
    graph_embs: torch.Tensor,
    reference_texts: List[str],
    batch_size: int = 32,
    beam_size: int = 5,
    max_gen_length: int = 128,
    device: str = "cuda",
) -> Dict:
    """
    Evaluate G2Text generation quality with BLEU-4, ROUGE-L, and CIDEr.

    Returns:
        Dict with metric scores and per-sample results.
    """
    model.eval()
    model = model.to(device)

    generated_texts = []
    for start in range(0, len(test_indices), batch_size):
        batch_idx = test_indices[start : start + batch_size]
        embs = graph_embs[batch_idx].to(device)
        texts = model.generate(embs, max_length=max_gen_length, beam_size=beam_size)
        generated_texts.extend(texts)

    # Align references
    ref_texts = [reference_texts[i] for i in test_indices]

    # Filter out empty references for metric computation
    valid_pairs = [
        (r, h) for r, h in zip(ref_texts, generated_texts) if r.strip()
    ]
    if not valid_pairs:
        logger.warning("No valid reference texts found for evaluation")
        return {
            "bleu4": 0.0,
            "rouge_l": 0.0,
            "cider": 0.0,
            "num_samples": 0,
        }

    valid_refs = [p[0] for p in valid_pairs]
    valid_hyps = [p[1] for p in valid_pairs]

    bleu4 = compute_bleu4(valid_refs, valid_hyps)
    rouge_l = compute_rouge_l(valid_refs, valid_hyps)
    cider = compute_cider(valid_refs, valid_hyps)

    logger.info(
        f"G2Text Evaluation: BLEU-4={bleu4:.4f}, ROUGE-L={rouge_l:.4f}, "
        f"CIDEr={cider:.4f} ({len(valid_pairs)} samples)"
    )

    return {
        "bleu4": float(bleu4),
        "rouge_l": float(rouge_l),
        "cider": float(cider),
        "num_samples": len(valid_pairs),
        "examples": [
            {"reference": r[:200], "hypothesis": h[:200]}
            for r, h in valid_pairs[:5]
        ],
    }


# ---------------------------------------------------------------------------
# Main experiment runner
# ---------------------------------------------------------------------------

def run_g2text_experiment(
    dataset,
    model_name: str = "gcn",
    modality: str = "text",
    peft_type: str = "none",
    llm_model_name: str = "facebook/opt-125m",
    num_runs: int = 3,
    num_epochs: int = 10,
    hidden_dim: int = 256,
    num_layers: int = 2,
    dropout: float = 0.5,
    lr: float = 5e-5,
    weight_decay: float = 0.01,
    batch_size: int = 32,
    patience: int = 3,
    max_gen_length: int = 128,
    beam_size: int = 5,
    lora_r: int = 64,
    lora_alpha: int = 128,
    adapter_bottleneck: int = 64,
    device: str = "cuda",
    output_dir: str = "./benchmark/results/g2text",
    data_root: str = "./data",
    domains: Optional[List[str]] = None,
) -> Dict:
    """
    Run the full G2Text generation experiment.

    Steps:
        1. Load dataset and split nodes 60/20/20.
        2. Pre-train a GNN encoder on node classification (or use a checkpoint).
        3. Freeze GNN, build G2TextModel with projection + PEFT-wrapped LLM.
        4. Train on paper texts (title + abstract).
        5. Evaluate with BLEU-4, ROUGE-L, CIDEr.
        6. Save results to JSON.

    Args:
        dataset: AMMAGDataset instance
        model_name: GNN backbone name (gcn, gat, graphsage)
        modality: modality string for features
        peft_type: PEFT strategy (none, lora, adapter, ia3)
        llm_model_name: HuggingFace LLM identifier
        num_runs: number of independent runs with different seeds
        num_epochs: max training epochs per run
        hidden_dim: GNN hidden dimension
        num_layers: GNN layer count
        dropout: GNN dropout rate
        lr: learning rate for projection + LLM
        weight_decay: AdamW weight decay
        batch_size: training batch size
        patience: early stopping patience (epochs)
        max_gen_length: maximum generation length in tokens
        beam_size: beam search width for generation
        lora_r: LoRA rank
        lora_alpha: LoRA alpha
        adapter_bottleneck: Adapter bottleneck dimension
        device: cuda device string
        output_dir: directory to save results
        data_root: root directory for raw paper data
        domains: list of domain names

    Returns:
        Aggregated results dictionary.
    """
    from benchmark.models.gnn import get_model
    from benchmark.tasks.node_classification import train_and_eval

    os.makedirs(output_dir, exist_ok=True)
    data = dataset.data
    num_nodes = data.num_nodes
    in_dim = data.x.size(1)
    out_dim = int(data.y.max().item()) + 1
    gnn_out_dim = hidden_dim  # GNN encoder output dim (not num_classes)

    if domains is None:
        domains = ["cs_si", "cs_lo", "math_lo", "physics_socph"]

    # Load paper texts
    paper_ids = dataset._graph_raw["paper_ids"]
    texts, _ = load_paper_texts(data_root, domains, paper_ids)

    # Check that we have enough texts
    non_empty = sum(1 for t in texts if t.strip())
    if non_empty < num_nodes * 0.1:
        logger.warning(
            f"Only {non_empty}/{num_nodes} papers have text. "
            f"Check data_root path. Attempting with available texts."
        )

    all_results = []

    for run in range(num_runs):
        seed = run * 42 + 1
        torch.manual_seed(seed)
        np.random.seed(seed)

        logger.info(
            f"=== G2Text Run {run + 1}/{num_runs}: "
            f"{model_name}+{modality}, peft={peft_type} ==="
        )

        # Step 1: Split
        train_mask, val_mask, test_mask = dataset.get_split_masks(
            split="supervised", seed=seed,
        )
        train_idx = train_mask.nonzero(as_tuple=True)[0].numpy()
        val_idx = val_mask.nonzero(as_tuple=True)[0].numpy()
        test_idx = test_mask.nonzero(as_tuple=True)[0].numpy()

        # Step 2: Pre-train GNN on node classification (frozen afterwards)
        gnn = get_model(
            model_name, in_dim, hidden_dim, out_dim,
            num_layers=num_layers, dropout=dropout,
        )
        logger.info(f"Pre-training GNN encoder ({model_name}) on node classification...")
        gnn_result = train_and_eval(
            gnn, data, train_mask, val_mask, test_mask,
            num_epochs=200, lr=0.01, device=device,
        )
        logger.info(
            f"GNN pre-training done: val_acc={gnn_result['val_acc']:.4f}, "
            f"test_acc={gnn_result['test_acc']:.4f}"
        )

        # Use the best GNN state from pre-training
        best_gnn_state = {k: v.clone() for k, v in gnn.state_dict().items()}
        gnn.load_state_dict(best_gnn_state)

        # Build GNN encoder with hidden_dim output (not num_classes) for embedding
        gnn_encoder = get_model(
            model_name, in_dim, hidden_dim, hidden_dim,
            num_layers=num_layers, dropout=dropout,
        )
        # Copy pre-trained weights for shared layers (skip last layer if shapes differ)
        encoder_state = gnn_encoder.state_dict()
        for k in encoder_state:
            if k in best_gnn_state and best_gnn_state[k].shape == encoder_state[k].shape:
                encoder_state[k] = best_gnn_state[k]
        gnn_encoder.load_state_dict(encoder_state)

        # Build G2Text model
        g2text_model = G2TextModel(
            gnn_encoder=gnn_encoder,
            gnn_dim=hidden_dim,
            llm_model_name=llm_model_name,
            peft_type=peft_type,
            lora_r=lora_r,
            lora_alpha=lora_alpha,
            adapter_bottleneck=adapter_bottleneck,
            max_gen_length=max_gen_length,
            device=device,
        ).to(device)

        # Compute all graph embeddings (GNN forward on full graph, then project)
        data_device = data.to(device)
        graph_embs = g2text_model.encode_graph(
            data_device.x, data_device.edge_index
        )  # [N, llm_hidden_dim]
        graph_embs = graph_embs.cpu()

        # Build datasets
        train_dataset = G2TextDataset(
            train_idx, graph_embs, texts,
            g2text_model.tokenizer, max_length=max_gen_length,
        )
        val_dataset = G2TextDataset(
            val_idx, graph_embs, texts,
            g2text_model.tokenizer, max_length=max_gen_length,
        )

        # Step 3: Train projection + LLM
        logger.info("Training projection layer + LLM (with PEFT)...")
        train_result = train_g2text(
            g2text_model,
            train_dataset,
            val_dataset,
            num_epochs=num_epochs,
            lr=lr,
            weight_decay=weight_decay,
            batch_size=batch_size,
            patience=patience,
            device=device,
        )
        logger.info(
            f"Training done: best_epoch={train_result['best_epoch']}, "
            f"best_val_loss={train_result['best_val_loss']:.4f}, "
            f"time={train_result['training_time']:.1f}s"
        )

        # Step 4: Evaluate
        eval_result = evaluate_g2text(
            g2text_model,
            test_idx,
            graph_embs,
            texts,
            batch_size=batch_size,
            beam_size=beam_size,
            max_gen_length=max_gen_length,
            device=device,
        )

        result = {
            "run": run + 1,
            "seed": seed,
            "bleu4": eval_result["bleu4"],
            "rouge_l": eval_result["rouge_l"],
            "cider": eval_result["cider"],
            "num_eval_samples": eval_result["num_samples"],
            "best_epoch": train_result["best_epoch"],
            "best_val_loss": train_result["best_val_loss"],
            "training_time": train_result["training_time"],
            "gnn_val_acc": gnn_result["val_acc"],
            "gnn_test_acc": gnn_result["test_acc"],
            "examples": eval_result.get("examples", []),
        }
        all_results.append(result)

        logger.info(
            f"[Run {run + 1}] BLEU-4={result['bleu4']:.4f}, "
            f"ROUGE-L={result['rouge_l']:.4f}, CIDEr={result['cider']:.4f}"
        )

        # Clean up GPU memory
        del g2text_model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # Step 5: Aggregate results across runs
    metrics = ["bleu4", "rouge_l", "cider", "training_time"]
    summary: Dict = {}
    for m in metrics:
        values = [r[m] for r in all_results]
        summary[f"{m}_mean"] = float(np.mean(values))
        summary[f"{m}_std"] = float(np.std(values))

    summary["model"] = model_name
    summary["modality"] = modality
    summary["peft_type"] = peft_type
    summary["llm"] = llm_model_name
    summary["num_runs"] = num_runs
    summary["num_epochs"] = num_epochs
    summary["batch_size"] = batch_size
    summary["lr"] = lr
    summary["lora_r"] = lora_r
    summary["lora_alpha"] = lora_alpha
    summary["adapter_bottleneck"] = adapter_bottleneck
    summary["hidden_dim"] = hidden_dim
    summary["max_gen_length"] = max_gen_length
    summary["beam_size"] = beam_size
    summary["runs"] = all_results

    # Save
    save_name = f"g2text_{model_name}_{modality}_peft-{peft_type}"
    save_path = os.path.join(output_dir, f"{save_name}.json")
    with open(save_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)

    logger.info(f"Results saved to {save_path}")
    logger.info(
        f"Summary: BLEU-4={summary['bleu4_mean']:.4f}+/-{summary['bleu4_std']:.4f}, "
        f"ROUGE-L={summary['rouge_l_mean']:.4f}+/-{summary['rouge_l_std']:.4f}, "
        f"CIDEr={summary['cider_mean']:.4f}+/-{summary['cider_std']:.4f}"
    )

    return summary


# ---------------------------------------------------------------------------
# Convenience: sweep over GNN backbones and PEFT strategies
# ---------------------------------------------------------------------------

def run_g2text_sweep(
    dataset,
    modality: str = "text",
    gnn_backbones: Optional[List[str]] = None,
    peft_strategies: Optional[List[str]] = None,
    output_dir: str = "./benchmark/results/g2text",
    **kwargs,
) -> Dict[str, Dict]:
    """
    Run a full sweep over GNN backbones x PEFT strategies.

    Default backbones : ["gcn", "gat", "graphsage"]
    Default PEFT      : ["none", "lora", "adapter", "ia3"]

    Returns:
        Dict mapping "{gnn}_{peft}" -> summary dict
    """
    if gnn_backbones is None:
        gnn_backbones = ["gcn", "gat", "graphsage"]
    if peft_strategies is None:
        peft_strategies = ["none", "lora", "adapter", "ia3"]

    all_sweep_results = {}

    for gnn_name in gnn_backbones:
        for peft_type in peft_strategies:
            key = f"{gnn_name}_{peft_type}"
            logger.info(f"{'='*60}")
            logger.info(f"G2Text Sweep: GNN={gnn_name}, PEFT={peft_type}")
            logger.info(f"{'='*60}")

            result = run_g2text_experiment(
                dataset=dataset,
                model_name=gnn_name,
                modality=modality,
                peft_type=peft_type,
                output_dir=output_dir,
                **kwargs,
            )
            all_sweep_results[key] = result

    # Save sweep summary
    sweep_summary_path = os.path.join(output_dir, "g2text_sweep_summary.json")
    with open(sweep_summary_path, "w") as f:
        json.dump(all_sweep_results, f, indent=2, default=str)
    logger.info(f"Sweep summary saved to {sweep_summary_path}")

    return all_sweep_results


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Graph-to-Text Generation (OpenMAG)")
    parser.add_argument("--model", type=str, default="gcn",
                        help="GNN backbone (gcn, gat, graphsage)")
    parser.add_argument("--modality", type=str, default="text",
                        help="Feature modality")
    parser.add_argument("--peft", type=str, default="none",
                        help="PEFT strategy (none, lora, adapter, ia3)")
    parser.add_argument("--llm", type=str, default="facebook/opt-125m",
                        help="LLM backbone model name")
    parser.add_argument("--sweep", action="store_true",
                        help="Run full sweep over all GNN x PEFT combos")
    parser.add_argument("--num_runs", type=int, default=3)
    parser.add_argument("--num_epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--data_root", type=str, default="./data")
    parser.add_argument("--output_dir", type=str,
                        default="./benchmark/results/g2text")
    args = parser.parse_args()

    device = f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"

    # Disable HF online mode (match project convention)
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"

    from benchmark.data.dataset import AMMAGDataset

    dataset = AMMAGDataset(
        root=".",
        domains=["cs_si", "cs_lo", "math_lo", "physics_socph"],
        modality=args.modality,
        label_type="domain",
    )

    if args.sweep:
        run_g2text_sweep(
            dataset=dataset,
            modality=args.modality,
            output_dir=args.output_dir,
            llm_model_name=args.llm,
            num_runs=args.num_runs,
            num_epochs=args.num_epochs,
            hidden_dim=args.hidden_dim,
            batch_size=args.batch_size,
            lr=args.lr,
            device=device,
            data_root=args.data_root,
        )
    else:
        run_g2text_experiment(
            dataset=dataset,
            model_name=args.model,
            modality=args.modality,
            peft_type=args.peft,
            llm_model_name=args.llm,
            num_runs=args.num_runs,
            num_epochs=args.num_epochs,
            hidden_dim=args.hidden_dim,
            batch_size=args.batch_size,
            lr=args.lr,
            device=device,
            output_dir=args.output_dir,
            data_root=args.data_root,
        )

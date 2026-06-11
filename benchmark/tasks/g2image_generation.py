"""
Graph-to-Image Generation Task (G2Image) - OpenMAG Benchmark
GNN encoder -> Graph-QFormer -> Stable Diffusion

Reproduces the G2Image task from the OpenMAG paper:
- GNN encodes the citation graph to produce node embeddings
- Graph-QFormer (2-layer lightweight transformer with cross-attention) maps
  graph-aware node embeddings to Stable Diffusion conditioning vectors
- Stable Diffusion generates paper figure images conditioned on graph structure
- Evaluated with CLIP-Score, FID, and optionally DINOv2-Score
- Neighbor ablation reproduces Table 2 (K={0,2,4,6,8})
"""
import os
import json
import time
import glob
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple
from PIL import Image
from loguru import logger


# ---------------------------------------------------------------------------
# 1. Graph-QFormer: Lightweight cross-attention bridge from GNN to SD
# ---------------------------------------------------------------------------

class GraphQFormer(nn.Module):
    """
    Lightweight transformer that maps GNN node embeddings to Stable Diffusion
    conditioning vectors via learnable queries and cross-attention.

    Architecture:
      - N learnable query embeddings (output tokens)
      - 2-layer Transformer with 8 attention heads
      - Cross-attention from queries to GNN embeddings (keys/values)
      - Output projection to SD UNet cross-attention dimension (768)
    """

    def __init__(
        self,
        gnn_dim: int = 256,
        sd_dim: int = 768,
        num_queries: int = 16,
        num_layers: int = 2,
        num_heads: int = 8,
        ff_dim: int = 2048,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.num_queries = num_queries
        self.sd_dim = sd_dim

        # Learnable query embeddings
        self.query_embed = nn.Parameter(torch.randn(1, num_queries, gnn_dim) * 0.02)

        # Project GNN embeddings to match query dimension for cross-attention
        self.kv_proj = nn.Linear(gnn_dim, gnn_dim)

        # Transformer decoder layers (cross-attention + self-attention)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=gnn_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerDecoder(
            decoder_layer, num_layers=num_layers
        )

        # Project output queries to SD conditioning dimension
        self.output_proj = nn.Sequential(
            nn.LayerNorm(gnn_dim),
            nn.Linear(gnn_dim, sd_dim),
        )

    def forward(self, gnn_embeddings: torch.Tensor) -> torch.Tensor:
        """
        Args:
            gnn_embeddings: (batch_size, gnn_dim) -- one per paper node

        Returns:
            conditioning: (batch_size, sd_dim) -- pooled conditioning vector
        """
        batch_size = gnn_embeddings.size(0)

        # Expand queries for batch
        queries = self.query_embed.expand(batch_size, -1, -1)  # (B, Q, D)

        # GNN embedding acts as memory for cross-attention
        memory = self.kv_proj(gnn_embeddings).unsqueeze(1)  # (B, 1, D)

        # Transformer decoder: queries attend to GNN memory
        out = self.transformer(queries, memory)  # (B, Q, D)

        # Pool across queries (mean) and project to SD dim
        pooled = out.mean(dim=1)  # (B, D)
        conditioning = self.output_proj(pooled)  # (B, sd_dim)

        return conditioning


# ---------------------------------------------------------------------------
# 2. Paper Image Dataset
# ---------------------------------------------------------------------------

class PaperImageDataset(torch.utils.data.Dataset):
    """
    Loads paper images paired with their GNN node indices.

    Each sample is one (node_idx, image_path) pair.  For papers with multiple
    images, all images are included as separate samples sharing the same node
    index (and therefore the same GNN embedding).
    """

    def __init__(
        self,
        paper_ids: List[str],
        domain_labels: torch.Tensor,
        domain_keys: List[str],
        data_root: str,
        node_indices: np.ndarray,
        resolution: int = 256,
    ):
        """
        Args:
            paper_ids: list of paper ID strings (full dataset)
            domain_labels: per-node domain label tensor
            domain_keys: domain name list ordered by domain index
            data_root: path to data/ directory
            node_indices: array of node indices to include (after split)
            resolution: output image resolution (square)
        """
        self.resolution = resolution
        self.samples = []  # list of (node_idx_in_full_graph, image_path)

        for idx in node_indices:
            pid = paper_ids[idx]
            dlabel = domain_labels[idx].item()
            dkey = domain_keys[dlabel]
            img_dir = os.path.join(data_root, f"{dkey}_papers", pid, "images")

            if not os.path.isdir(img_dir):
                continue

            for img_file in sorted(os.listdir(img_dir)):
                if img_file.lower().endswith((".png", ".jpg", ".jpeg")):
                    self.samples.append((int(idx), os.path.join(img_dir, img_file)))

        logger.info(
            f"PaperImageDataset: {len(self.samples)} image-node pairs "
            f"from {len(set(s[0] for s in self.samples))} papers"
        )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        node_idx, img_path = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        # Resize to target resolution
        image = image.resize((self.resolution, self.resolution), Image.BILINEAR)
        # Convert to tensor [0, 1]
        img_tensor = torch.from_numpy(np.array(image)).permute(2, 0, 1).float() / 255.0
        return node_idx, img_tensor


# ---------------------------------------------------------------------------
# 3. Metrics
# ---------------------------------------------------------------------------

class CLIPScoreMetric:
    """CLIP-Score: cosine similarity between generated-image and text-title embeddings."""

    def __init__(self, device: str = "cuda"):
        from transformers import CLIPModel, CLIPProcessor
        self.device = device
        self.model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(device)
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
        self.model.eval()

    @torch.no_grad()
    def compute_image_features(self, images: List[Image.Image]) -> torch.Tensor:
        """Compute normalized CLIP image features for a list of PIL images."""
        inputs = self.processor(images=images, return_tensors="pt", padding=True).to(self.device)
        features = self.model.get_image_features(**inputs)
        return F.normalize(features, dim=-1)

    @torch.no_grad()
    def compute_text_features(self, texts: List[str]) -> torch.Tensor:
        """Compute normalized CLIP text features."""
        inputs = self.processor(text=texts, return_tensors="pt", padding=True, truncation=True).to(self.device)
        features = self.model.get_text_features(**inputs)
        return F.normalize(features, dim=-1)

    def compute_clip_score(
        self, gen_images: List[Image.Image], text_titles: List[str]
    ) -> float:
        """Mean cosine similarity between generated images and paper titles."""
        img_feats = self.compute_image_features(gen_images)
        txt_feats = self.compute_text_features(text_titles)
        scores = (img_feats * txt_feats).sum(dim=-1)
        return scores.mean().item()


class FIDMetric:
    """
    Frechet Inception Distance between generated and real image distributions.

    Uses Inception v3 to extract 2048-d feature vectors, then computes the
    Frechet distance between two multivariate Gaussians fitted to the
    features of each distribution.
    """

    def __init__(self, device: str = "cuda"):
        self.device = device
        from torchvision.models import inception_v3, Inception_V3_Weights
        weights = Inception_V3_Weights.DEFAULT
        self.model = inception_v3(weights=weights).to(device)
        self.model.eval()
        # Remove final classification layer; output is pool3 (N x 2048)
        self.model.fc = nn.Identity()
        self.transform = weights.transforms()

    @torch.no_grad()
    def _extract_features(self, images: List[Image.Image], batch_size: int = 32) -> np.ndarray:
        """Extract Inception pool-3 features from PIL images."""
        all_feats = []
        for i in range(0, len(images), batch_size):
            batch_imgs = images[i : i + batch_size]
            tensors = []
            for img in batch_imgs:
                if not isinstance(img, Image.Image):
                    img = Image.fromarray(img)
                t = self.transform(img)
                tensors.append(t)
            batch = torch.stack(tensors).to(self.device)
            feats = self.model(batch).cpu().numpy()
            all_feats.append(feats)
        return np.concatenate(all_feats, axis=0)

    def compute_fid(
        self,
        real_images: List[Image.Image],
        gen_images: List[Image.Image],
        batch_size: int = 32,
    ) -> float:
        """Compute FID between real and generated image sets."""
        real_feats = self._extract_features(real_images, batch_size)
        gen_feats = self._extract_features(gen_images, batch_size)

        mu_r, sigma_r = np.mean(real_feats, axis=0), np.cov(real_feats, rowvar=False)
        mu_g, sigma_g = np.mean(gen_feats, axis=0), np.cov(gen_feats, rowvar=False)

        from scipy.linalg import sqrtm
        diff = mu_r - mu_g
        covmean = sqrtm(sigma_r @ sigma_g)
        # Numerical stability
        if np.iscomplexobj(covmean):
            covmean = covmean.real
        fid = diff @ diff + np.trace(sigma_r + sigma_g - 2 * covmean)
        return float(fid)


class DINOv2ScoreMetric:
    """
    DINOv2-Score: cosine similarity between generated-image and real-image
    DINOv2 features.  Falls back gracefully if DINOv2 is unavailable.
    """

    def __init__(self, device: str = "cuda"):
        self.device = device
        self.available = False
        try:
            from transformers import AutoModel, AutoImageProcessor
            self.processor = AutoImageProcessor.from_pretrained("facebook/dinov2-base")
            self.model = AutoModel.from_pretrained("facebook/dinov2-base").to(device)
            self.model.eval()
            self.available = True
            logger.info("DINOv2 loaded successfully")
        except Exception as e:
            logger.warning(f"DINOv2 unavailable ({e}), will fall back to CLIP-Score")

    @torch.no_grad()
    def compute_score(
        self, gen_images: List[Image.Image], real_images: List[Image.Image]
    ) -> Optional[float]:
        if not self.available:
            return None

        def _feats(imgs):
            inputs = self.processor(images=imgs, return_tensors="pt").to(self.device)
            out = self.model(**inputs)
            return F.normalize(out.last_hidden_state[:, 0], dim=-1)

        gen_f = _feats(gen_images)
        real_f = _feats(real_images)
        return (gen_f * real_f).sum(dim=-1).mean().item()


# ---------------------------------------------------------------------------
# 4. Training
# ---------------------------------------------------------------------------

def build_sd_pipeline(device: str = "cuda"):
    """Load Stable Diffusion v1.5 in fp16 for single-GPU efficiency."""
    from diffusers import StableDiffusionPipeline

    pipe = StableDiffusionPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        torch_dtype=torch.float16,
        safety_checker=None,
    )
    pipe = pipe.to(device)
    pipe.set_progress_bar_config(disable=True)
    return pipe


def apply_lora_to_unet(unet, lora_rank: int = 4):
    """Apply LoRA to SD UNet cross-attention projection layers."""
    from peft import LoraConfig, get_peft_model

    lora_config = LoraConfig(
        r=lora_rank,
        lora_alpha=lora_rank,
        target_modules=["to_q", "to_k", "to_v", "to_out.0"],
        lora_dropout=0.0,
    )
    unet = get_peft_model(unet, lora_config)
    unet.print_trainable_parameters()
    return unet


def train_g2image(
    gnn_model: nn.Module,
    graph_qformer: GraphQFormer,
    sd_pipeline,
    data,                         # PyG Data object
    paper_ids: List[str],
    domain_labels: torch.Tensor,
    domain_keys: List[str],
    data_root: str,
    train_indices: np.ndarray,
    val_indices: np.ndarray,
    device: str = "cuda",
    num_epochs: int = 20,
    lr: float = 1e-4,
    lora_rank: int = 4,
    batch_size: int = 4,
    resolution: int = 256,
    clip_feature_loss_weight: float = 0.5,
    num_train_timesteps: int = 1000,
    use_lora: bool = True,
    patience: int = 5,
    save_dir: str = None,
):
    """
    Train the G2Image pipeline: frozen GNN -> trainable GraphQFormer -> SD with LoRA.

    Two losses:
      1. SD denoising loss (standard diffusion MSE on predicted noise)
      2. CLIP feature MSE between generated and real image features

    Args:
        gnn_model: pre-trained GNN encoder (will be frozen)
        graph_qformer: GraphQFormer module (trainable)
        sd_pipeline: StableDiffusionPipeline
        data: PyG Data with x, edge_index
        paper_ids: list of paper IDs
        domain_labels: domain label per node
        domain_keys: list of domain name strings
        data_root: path to data/ directory
        train_indices: node indices for training
        val_indices: node indices for validation
        device: cuda device string
        num_epochs: max training epochs
        lr: learning rate for QFormer + LoRA params
        lora_rank: LoRA rank for UNet fine-tuning
        batch_size: images per gradient step
        resolution: image resolution
        clip_feature_loss_weight: weight for CLIP feature alignment loss
        num_train_timesteps: diffusion timesteps
        use_lora: whether to fine-tune UNet with LoRA
        patience: early stopping patience (in epochs)
        save_dir: directory to save best model checkpoint

    Returns:
        trained graph_qformer, sd_pipeline
    """
    from diffusers import DDPMScheduler
    from transformers import CLIPModel, CLIPProcessor

    os.makedirs(save_dir or ".", exist_ok=True)

    # ---- Freeze GNN ----
    gnn_model = gnn_model.to(device).eval()
    for p in gnn_model.parameters():
        p.requires_grad = False

    # ---- Pre-compute GNN embeddings for all nodes (one forward pass) ----
    with torch.no_grad():
        gnn_out = gnn_model(data.x.to(device), data.edge_index.to(device))  # (N, hidden)
    gnn_out = gnn_out.detach()  # (N, hidden_dim)

    # ---- Setup GraphQFormer (trainable) ----
    graph_qformer = graph_qformer.to(device)

    # ---- Setup SD UNet (optionally LoRA) ----
    unet = sd_pipeline.unet
    vae = sd_pipeline.vae
    text_encoder = sd_pipeline.text_encoder
    noise_scheduler = sd_pipeline.scheduler

    if use_lora:
        unet = apply_lora_to_unet(unet, lora_rank=lora_rank)
        unet = unet.to(device)

    # Freeze VAE and text encoder
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)

    # ---- CLIP for feature loss ----
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(device)
    clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
    clip_model.eval()
    for p in clip_model.parameters():
        p.requires_grad = False

    # ---- Noise scheduler ----
    train_scheduler = DDPMScheduler.from_config(sd_pipeline.scheduler.config)
    train_scheduler.set_timesteps(num_train_timesteps)

    # ---- Build image dataset for training ----
    train_dataset = PaperImageDataset(
        paper_ids=paper_ids,
        domain_labels=domain_labels,
        domain_keys=domain_keys,
        data_root=data_root,
        node_indices=train_indices,
        resolution=resolution,
    )

    # Filter to only include nodes whose GNN embeddings are valid
    valid_node_set = set(train_indices.tolist()) if isinstance(train_indices, np.ndarray) else set(train_indices)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=2, pin_memory=True, drop_last=True,
    )

    # ---- Optimizer: GraphQFormer + LoRA params ----
    trainable_params = list(graph_qformer.parameters())
    if use_lora:
        trainable_params += [p for p in unet.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=lr)
    scaler = torch.cuda.amp.GradScaler(enabled=True)

    # ---- Training loop ----
    best_val_loss = float("inf")
    patience_counter = 0
    epoch_losses = []

    logger.info(f"G2Image training: {len(train_dataset)} samples, {num_epochs} epochs, "
                f"batch_size={batch_size}, lr={lr}")

    for epoch in range(1, num_epochs + 1):
        graph_qformer.train()
        if use_lora:
            unet.train()

        total_loss = 0.0
        total_denoise_loss = 0.0
        total_clip_loss = 0.0
        num_batches = 0

        for node_idxs, real_imgs in train_loader:
            # real_imgs: (B, 3, H, W)
            real_imgs = real_imgs.to(device)
            node_idxs = node_idxs.to(device)

            # Get GNN embeddings for this batch
            gnn_embs = gnn_out[node_idxs]  # (B, hidden_dim)

            # Graph-QFormer: map to SD conditioning vectors
            cond_vectors = graph_qformer(gnn_embs)  # (B, 768)

            # ---- Encode real images to latent space ----
            # VAE encode expects float16 or float32
            with torch.no_grad():
                real_imgs_f16 = real_imgs.half()
                latents = vae.encode(real_imgs_f16).latent_dist.sample()
                latents = latents * vae.config.scaling_factor
                latents = latents.float()  # back to fp32 for noise

            # ---- Add noise ----
            noise = torch.randn_like(latents)
            timesteps = torch.randint(
                0, num_train_timesteps, (latents.size(0),), device=device
            ).long()
            noisy_latents = train_scheduler.add_noise(latents, noise, timesteps)

            # ---- UNet forward with graph-conditioned embeddings ----
            # Expand conditioning to match UNet cross-attention expected shape: (B, seq_len, dim)
            encoder_hidden_states = cond_vectors.unsqueeze(1)  # (B, 1, 768)
            encoder_hidden_states = encoder_hidden_states.half()

            with torch.cuda.amp.autocast(enabled=True):
                noise_pred = unet(
                    noisy_latents.half(),
                    timesteps,
                    encoder_hidden_states=encoder_hidden_states,
                ).sample.float()

                # Denoising loss
                denoise_loss = F.mse_loss(noise_pred, noise)

                # CLIP feature alignment loss
                # Decode predicted clean latents
                with torch.no_grad():
                    # Get predicted clean sample via scheduler step
                    alpha_t = train_scheduler.alphas_cumprod[timesteps].view(-1, 1, 1, 1)
                    pred_clean = (noisy_latents - (1 - alpha_t).sqrt() * noise_pred) / alpha_t.sqrt().clamp(min=1e-8)
                    pred_clean = pred_clean.half()
                    decoded = vae.decode(pred_clean / vae.config.scaling_factor).sample
                    decoded_imgs = decoded.float()

                # Convert decoded images to [0,1] for CLIP
                decoded_imgs = (decoded_imgs / 2 + 0.5).clamp(0, 1)
                real_imgs_01 = real_imgs.clamp(0, 1)

                # CLIP features
                with torch.no_grad():
                    # Resize to CLIP expected size (224)
                    decoded_resized = F.interpolate(decoded_imgs, size=(224, 224), mode="bilinear", align_corners=False)
                    real_resized = F.interpolate(real_imgs_01, size=(224, 224), mode="bilinear", align_corners=False)

                    # Normalize with CLIP stats
                    clip_mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1).to(device)
                    clip_std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1).to(device)
                    decoded_norm = (decoded_resized - clip_mean) / clip_std
                    real_norm = (real_resized - clip_mean) / clip_std

                    gen_features = clip_model.get_image_features(pixel_values=decoded_norm)
                    real_features = clip_model.get_image_features(pixel_values=real_norm)
                    gen_features = F.normalize(gen_features, dim=-1)
                    real_features = F.normalize(real_features, dim=-1)

                clip_loss = F.mse_loss(gen_features, real_features.detach())

                # Total loss
                loss = denoise_loss + clip_feature_loss_weight * clip_loss

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()
            total_denoise_loss += denoise_loss.item()
            total_clip_loss += clip_loss.item()
            num_batches += 1

        avg_loss = total_loss / max(num_batches, 1)
        avg_denoise = total_denoise_loss / max(num_batches, 1)
        avg_clip = total_clip_loss / max(num_batches, 1)
        epoch_losses.append(avg_loss)

        logger.info(
            f"Epoch {epoch}/{num_epochs}: loss={avg_loss:.4f} "
            f"(denoise={avg_denoise:.4f}, clip={avg_clip:.4f})"
        )

        # Validation: use a small subset to track loss
        if len(val_indices) > 0:
            val_loss = _validate_g2image(
                gnn_out, graph_qformer, unet, vae, train_scheduler,
                paper_ids, domain_labels, domain_keys, data_root,
                val_indices, device, batch_size, num_train_timesteps,
            )
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                if save_dir:
                    torch.save(
                        {
                            "graph_qformer": graph_qformer.state_dict(),
                            "epoch": epoch,
                            "val_loss": val_loss,
                        },
                        os.path.join(save_dir, "best_qformer.pt"),
                    )
                    if use_lora:
                        unet.save_pretrained(os.path.join(save_dir, "best_unet_lora"))
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    logger.info(f"Early stopping at epoch {epoch}")
                    break

    # Load best model if available
    if save_dir and os.path.exists(os.path.join(save_dir, "best_qformer.pt")):
        ckpt = torch.load(os.path.join(save_dir, "best_qformer.pt"), map_location=device)
        graph_qformer.load_state_dict(ckpt["graph_qformer"])
        logger.info(f"Loaded best QFormer from epoch {ckpt['epoch']} (val_loss={ckpt['val_loss']:.4f})")

    return graph_qformer, sd_pipeline


def _validate_g2image(
    gnn_out, graph_qformer, unet, vae, scheduler,
    paper_ids, domain_labels, domain_keys, data_root,
    val_indices, device, batch_size, num_timesteps,
):
    """Run a quick validation pass and return average loss."""
    graph_qformer.eval()
    if hasattr(unet, "eval"):
        unet.eval()

    val_dataset = PaperImageDataset(
        paper_ids=paper_ids,
        domain_labels=domain_labels,
        domain_keys=domain_keys,
        data_root=data_root,
        node_indices=val_indices,
        resolution=256,
    )
    # Limit validation samples for speed
    max_val_samples = min(200, len(val_dataset))
    if max_val_samples == 0:
        return 0.0

    subset_indices = np.random.choice(len(val_dataset), max_val_samples, replace=False)
    val_subset = torch.utils.data.Subset(val_dataset, subset_indices)
    val_loader = torch.utils.data.DataLoader(
        val_subset, batch_size=batch_size, shuffle=False, num_workers=1,
    )

    total_loss = 0.0
    count = 0

    with torch.no_grad():
        for node_idxs, real_imgs in val_loader:
            real_imgs = real_imgs.to(device)
            node_idxs = node_idxs.to(device)

            gnn_embs = gnn_out[node_idxs]
            cond_vectors = graph_qformer(gnn_embs)
            latents = vae.encode(real_imgs.half()).latent_dist.sample() * vae.config.scaling_factor
            latents = latents.float()

            noise = torch.randn_like(latents)
            timesteps = torch.randint(0, num_timesteps, (latents.size(0),), device=device).long()
            noisy_latents = scheduler.add_noise(latents, noise, timesteps)

            encoder_hidden_states = cond_vectors.unsqueeze(1).half()
            noise_pred = unet(
                noisy_latents.half(), timesteps, encoder_hidden_states=encoder_hidden_states,
            ).sample.float()

            loss = F.mse_loss(noise_pred, noise)
            total_loss += loss.item()
            count += 1

    return total_loss / max(count, 1)


# ---------------------------------------------------------------------------
# 5. Image Generation
# ---------------------------------------------------------------------------

def generate_images(
    gnn_out: torch.Tensor,
    graph_qformer: nn.Module,
    sd_pipeline,
    node_indices: np.ndarray,
    device: str = "cuda",
    num_inference_steps: int = 50,
    guidance_scale: float = 7.5,
    batch_size: int = 8,
    resolution: int = 256,
    output_dir: str = None,
) -> Tuple[List[Image.Image], List[int]]:
    """
    Generate images for specified nodes using the trained pipeline.

    Args:
        gnn_out: pre-computed GNN embeddings (N, hidden_dim)
        graph_qformer: trained GraphQFormer
        sd_pipeline: SD pipeline (with optional LoRA)
        node_indices: which nodes to generate images for
        device: cuda device
        num_inference_steps: DDIM steps
        guidance_scale: classifier-free guidance scale
        batch_size: generation batch size
        resolution: output image resolution
        output_dir: if set, save images to this directory

    Returns:
        (generated_images, node_index_list)
    """
    graph_qformer.eval()
    sd_pipeline.unet.eval()

    all_images = []
    all_node_idxs = []

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    with torch.no_grad():
        for start in range(0, len(node_indices), batch_size):
            batch_idxs = node_indices[start : start + batch_size]
            batch_t = torch.tensor(batch_idxs, dtype=torch.long, device=device)

            gnn_embs = gnn_out[batch_t]
            cond_vectors = graph_qformer(gnn_embs)  # (B, 768)

            # Prepare encoder_hidden_states for SD
            # SD expects (B, seq_len, cross_attention_dim)
            # Use conditioning as a "text embedding" surrogate
            encoder_hidden_states = cond_vectors.unsqueeze(1).half()  # (B, 1, 768)

            # Generate latents
            latents = torch.randn(
                (len(batch_idxs), sd_pipeline.unet.config.in_channels, resolution // 8, resolution // 8),
                device=device,
                dtype=torch.float16,
            )

            # Scale latents
            latents = latents * sd_pipeline.scheduler.init_noise_sigma

            # Denoise
            sd_pipeline.scheduler.set_timesteps(num_inference_steps)
            timesteps = sd_pipeline.scheduler.timesteps

            for t in timesteps:
                latent_model_input = latents
                noise_pred = sd_pipeline.unet(
                    latent_model_input,
                    t,
                    encoder_hidden_states=encoder_hidden_states,
                ).sample

                # Classifier-free guidance with null embedding
                # For simplicity, use the conditioning directly (guidance_scale=1 effectively)
                if guidance_scale > 1.0:
                    null_hidden = torch.zeros_like(encoder_hidden_states)
                    noise_pred_uncond = sd_pipeline.unet(
                        latent_model_input, t, encoder_hidden_states=null_hidden,
                    ).sample
                    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred - noise_pred_uncond)

                latents = sd_pipeline.scheduler.step(noise_pred, t, latents).prev_sample

            # Decode latents to images
            latents = latents / sd_pipeline.vae.config.scaling_factor
            images = sd_pipeline.vae.decode(latents).sample

            # Convert to PIL
            for i, img_tensor in enumerate(images):
                img = img_tensor.float().cpu()
                img = (img / 2 + 0.5).clamp(0, 1)
                img = img.permute(1, 2, 0).numpy()
                img = (img * 255).astype(np.uint8)
                pil_img = Image.fromarray(img)

                all_images.append(pil_img)
                all_node_idxs.append(int(batch_idxs[i]))

                if output_dir:
                    pil_img.save(os.path.join(output_dir, f"node_{batch_idxs[i]}.png"))

    return all_images, all_node_idxs


# ---------------------------------------------------------------------------
# 6. Evaluation
# ---------------------------------------------------------------------------

def evaluate_g2image(
    generated_images: List[Image.Image],
    real_images: List[Image.Image],
    text_titles: List[str],
    node_indices: List[int],
    paper_ids: List[str],
    device: str = "cuda",
) -> Dict:
    """
    Evaluate generated images against ground truth.

    Metrics:
      - CLIP-Score: cos-sim(gen_image_feat, title_text_feat)
      - FID: Frechet Inception Distance between gen and real distributions
      - DINOv2-Score: cos-sim(gen_image_feat, real_image_feat) via DINOv2 (optional)
    """
    results = {}

    # CLIP-Score
    logger.info("Computing CLIP-Score...")
    clip_metric = CLIPScoreMetric(device=device)
    clip_score = clip_metric.compute_clip_score(generated_images, text_titles)
    results["clip_score"] = float(clip_score)
    logger.info(f"  CLIP-Score: {clip_score:.4f}")

    # FID
    logger.info("Computing FID...")
    fid_metric = FIDMetric(device=device)
    fid = fid_metric.compute_fid(real_images, generated_images)
    results["fid"] = float(fid)
    logger.info(f"  FID: {fid:.4f}")

    # DINOv2-Score (optional)
    logger.info("Computing DINOv2-Score...")
    dino_metric = DINOv2ScoreMetric(device=device)
    dino_score = dino_metric.compute_score(generated_images, real_images)
    if dino_score is not None:
        results["dinov2_score"] = float(dino_score)
        logger.info(f"  DINOv2-Score: {dino_score:.4f}")
    else:
        # Fall back to CLIP-Score as image-image similarity
        logger.info("  DINOv2 unavailable, using CLIP image-image similarity as fallback")
        clip_img_score = _clip_image_similarity(generated_images, real_images, clip_metric)
        results["clip_image_score_fallback"] = float(clip_img_score)
        logger.info(f"  CLIP image-image score: {clip_img_score:.4f}")

    return results


def _clip_image_similarity(
    gen_images: List[Image.Image],
    real_images: List[Image.Image],
    clip_metric: CLIPScoreMetric,
) -> float:
    """CLIP-based image-to-image similarity as DINOv2 fallback."""
    gen_feats = clip_metric.compute_image_features(gen_images)
    real_feats = clip_metric.compute_image_features(real_images)
    scores = (gen_feats * real_feats).sum(dim=-1)
    return scores.mean().item()


# ---------------------------------------------------------------------------
# 7. Neighbor Ablation (Table 2 reproduction)
# ---------------------------------------------------------------------------

def run_neighbor_ablation_g2image(
    dataset,
    model_name: str,
    modality: str,
    data_root: str,
    paper_ids: List[str],
    domain_labels: torch.Tensor,
    domain_keys: List[str],
    neighbor_counts: List[int] = None,
    num_runs: int = 3,
    num_epochs: int = 15,
    hidden_dim: int = 256,
    num_layers: int = 2,
    dropout: float = 0.5,
    lr: float = 1e-4,
    batch_size: int = 4,
    device: str = "cuda",
    output_dir: str = "./benchmark/results/g2image",
):
    """
    Neighbor ablation for G2Image (OpenMAG Table 2).

    Tests with K={0, 2, 4, 6, 8} neighbors to measure the impact of graph
    structure on image generation quality.

    K=0 means no edges (GNN degrades to MLP, no message passing).
    """
    from benchmark.models.gnn import get_model
    from benchmark.tasks.neighbor_ablation import truncate_neighbors

    if neighbor_counts is None:
        neighbor_counts = [0, 2, 4, 6, 8]

    os.makedirs(output_dir, exist_ok=True)
    data = dataset.data

    # Collect paper titles (used for CLIP-Score)
    paper_titles = _load_paper_titles(data_root, paper_ids, domain_labels, domain_keys)

    all_results = {}

    for k in neighbor_counts:
        k_results = []
        for run in range(num_runs):
            seed = run * 42 + 1
            torch.manual_seed(seed)
            np.random.seed(seed)

            # Truncate edges
            if k == 0:
                truncated_edges = torch.zeros((2, 0), dtype=data.edge_index.dtype)
            else:
                truncated_edges = truncate_neighbors(
                    data.edge_index, data.num_nodes, k
                )

            # Create modified data
            data_k = data.clone()
            data_k.edge_index = truncated_edges

            # Split
            train_mask, val_mask, test_mask = dataset.get_split_masks(
                split="supervised", seed=seed
            )
            train_idx = train_mask.nonzero(as_tuple=True)[0].numpy()
            val_idx = val_mask.nonzero(as_tuple=True)[0].numpy()
            test_idx = test_mask.nonzero(as_tuple=True)[0].numpy()

            # Filter to papers with images
            train_idx = _filter_to_papers_with_images(
                train_idx, paper_ids, domain_labels, domain_keys, data_root
            )
            val_idx = _filter_to_papers_with_images(
                val_idx, paper_ids, domain_labels, domain_keys, data_root
            )
            test_idx = _filter_to_papers_with_images(
                test_idx, paper_ids, domain_labels, domain_keys, data_root
            )

            if len(test_idx) == 0:
                logger.warning(f"K={k}, run={run}: no test papers with images, skipping")
                continue

            # Build model
            in_dim = data_k.x.size(1)
            gnn_model = get_model(
                model_name, in_dim, hidden_dim, hidden_dim,
                num_layers=num_layers, dropout=dropout,
            )

            graph_qformer = GraphQFormer(
                gnn_dim=hidden_dim, sd_dim=768,
                num_queries=16, num_layers=2, num_heads=8,
            )

            sd_pipeline = build_sd_pipeline(device)

            save_dir = os.path.join(output_dir, f"ablation_k{k}_run{run}")

            # Train
            graph_qformer, sd_pipeline = train_g2image(
                gnn_model=gnn_model,
                graph_qformer=graph_qformer,
                sd_pipeline=sd_pipeline,
                data=data_k,
                paper_ids=paper_ids,
                domain_labels=domain_labels,
                domain_keys=domain_keys,
                data_root=data_root,
                train_indices=train_idx,
                val_indices=val_idx,
                device=device,
                num_epochs=num_epochs,
                lr=lr,
                batch_size=batch_size,
                save_dir=save_dir,
            )

            # Pre-compute GNN embeddings
            gnn_model.eval()
            with torch.no_grad():
                gnn_out = gnn_model(data_k.x.to(device), data_k.edge_index.to(device))
            gnn_out = gnn_out.detach()

            # Generate images for test set (limit for efficiency)
            test_gen_idx = test_idx[:200] if len(test_idx) > 200 else test_idx
            gen_images, gen_node_idxs = generate_images(
                gnn_out=gnn_out,
                graph_qformer=graph_qformer,
                sd_pipeline=sd_pipeline,
                node_indices=test_gen_idx,
                device=device,
                batch_size=batch_size,
                resolution=256,
                output_dir=os.path.join(save_dir, "generated"),
            )

            # Collect real images and titles for evaluation
            real_images, text_titles = _collect_eval_data(
                gen_node_idxs, paper_ids, paper_titles, domain_labels, domain_keys, data_root
            )

            if len(gen_images) > 0 and len(real_images) > 0:
                metrics = evaluate_g2image(
                    generated_images=gen_images,
                    real_images=real_images,
                    text_titles=text_titles,
                    node_indices=gen_node_idxs,
                    paper_ids=paper_ids,
                    device=device,
                )
                k_results.append(metrics)

            # Free GPU memory
            del gnn_model, graph_qformer, sd_pipeline
            torch.cuda.empty_cache()

        if k_results:
            summary = {}
            for metric_key in k_results[0]:
                vals = [r[metric_key] for r in k_results if metric_key in r]
                if vals:
                    summary[f"{metric_key}_mean"] = float(np.mean(vals))
                    summary[f"{metric_key}_std"] = float(np.std(vals))
            all_results[f"k_{k}"] = summary
            logger.info(f"Neighbor ablation K={k}: {summary}")

    # Save
    ablation_path = os.path.join(output_dir, f"neighbor_ablation_{model_name}_{modality}.json")
    with open(ablation_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    logger.info(f"Neighbor ablation results saved to {ablation_path}")

    return all_results


# ---------------------------------------------------------------------------
# 8. Main entry point
# ---------------------------------------------------------------------------

def run_g2image(
    dataset,
    model_name: str = "gcn",
    modality: str = "text",
    data_root: str = "./data",
    num_runs: int = 3,
    num_epochs: int = 20,
    hidden_dim: int = 256,
    num_layers: int = 2,
    dropout: float = 0.5,
    lr: float = 1e-4,
    batch_size: int = 4,
    lora_rank: int = 4,
    device: str = "cuda",
    output_dir: str = "./benchmark/results/g2image",
    resolution: int = 256,
):
    """
    Run the full G2Image generation experiment.

    For each of num_runs:
      1. Split nodes 60/20/20
      2. Filter to papers with images
      3. Train GNN (frozen) -> GraphQFormer -> SD (LoRA)
      4. Generate images for test set
      5. Evaluate with CLIP-Score, FID, DINOv2-Score

    Also runs neighbor ablation (K={0,2,4,6,8}) for Table 2.
    """
    from benchmark.models.gnn import get_model

    os.makedirs(output_dir, exist_ok=True)
    data = dataset.data

    # Load graph metadata
    graph_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "processed", "graph_data.pt"
    )
    graph_raw = torch.load(graph_path, map_location="cpu")
    paper_ids = graph_raw["paper_ids"]
    domain_labels = graph_raw["labels"]
    domain_keys = ["cs_si", "cs_lo", "math_lo", "physics_socph"]

    # Load paper titles
    paper_titles = _load_paper_titles(data_root, paper_ids, domain_labels, domain_keys)

    in_dim = data.x.size(1)
    all_run_results = []

    for run in range(num_runs):
        seed = run * 42 + 1
        torch.manual_seed(seed)
        np.random.seed(seed)

        logger.info(f"\n{'='*60}")
        logger.info(f"G2Image Run {run+1}/{num_runs} | model={model_name}, modality={modality}")
        logger.info(f"{'='*60}")

        # Split
        train_mask, val_mask, test_mask = dataset.get_split_masks(
            split="supervised", seed=seed
        )
        train_idx = train_mask.nonzero(as_tuple=True)[0].numpy()
        val_idx = val_mask.nonzero(as_tuple=True)[0].numpy()
        test_idx = test_mask.nonzero(as_tuple=True)[0].numpy()

        # Filter to papers that have at least 1 image
        train_idx = _filter_to_papers_with_images(
            train_idx, paper_ids, domain_labels, domain_keys, data_root
        )
        val_idx = _filter_to_papers_with_images(
            val_idx, paper_ids, domain_labels, domain_keys, data_root
        )
        test_idx = _filter_to_papers_with_images(
            test_idx, paper_ids, domain_labels, domain_keys, data_root
        )

        logger.info(
            f"Split: train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)} "
            f"(papers with images)"
        )

        # Build GNN model
        gnn_model = get_model(
            model_name, in_dim, hidden_dim, hidden_dim,
            num_layers=num_layers, dropout=dropout,
        )

        # Build Graph-QFormer
        graph_qformer = GraphQFormer(
            gnn_dim=hidden_dim,
            sd_dim=768,
            num_queries=16,
            num_layers=2,
            num_heads=8,
        )

        # Build SD pipeline
        sd_pipeline = build_sd_pipeline(device)

        save_dir = os.path.join(output_dir, f"run{run}_{model_name}_{modality}")

        # Train
        start_time = time.time()
        graph_qformer, sd_pipeline = train_g2image(
            gnn_model=gnn_model,
            graph_qformer=graph_qformer,
            sd_pipeline=sd_pipeline,
            data=data,
            paper_ids=paper_ids,
            domain_labels=domain_labels,
            domain_keys=domain_keys,
            data_root=data_root,
            train_indices=train_idx,
            val_indices=val_idx,
            device=device,
            num_epochs=num_epochs,
            lr=lr,
            batch_size=batch_size,
            lora_rank=lora_rank,
            save_dir=save_dir,
        )
        training_time = time.time() - start_time

        # Pre-compute GNN embeddings for generation
        gnn_model.eval()
        with torch.no_grad():
            gnn_out = gnn_model(data.x.to(device), data.edge_index.to(device))
        gnn_out = gnn_out.detach()

        # Generate images for test set (cap at 500 for evaluation speed)
        test_gen_idx = test_idx[:500] if len(test_idx) > 500 else test_idx
        gen_images, gen_node_idxs = generate_images(
            gnn_out=gnn_out,
            graph_qformer=graph_qformer,
            sd_pipeline=sd_pipeline,
            node_indices=test_gen_idx,
            device=device,
            batch_size=batch_size,
            resolution=resolution,
            output_dir=os.path.join(save_dir, "generated"),
        )

        # Collect real images and titles for evaluation
        real_images, text_titles = _collect_eval_data(
            gen_node_idxs, paper_ids, paper_titles, domain_labels, domain_keys, data_root
        )

        # Evaluate
        if len(gen_images) > 0 and len(real_images) > 0:
            metrics = evaluate_g2image(
                generated_images=gen_images,
                real_images=real_images,
                text_titles=text_titles,
                node_indices=gen_node_idxs,
                paper_ids=paper_ids,
                device=device,
            )
            metrics["training_time"] = training_time
            metrics["num_generated"] = len(gen_images)
            metrics["num_test_papers"] = len(test_gen_idx)
            all_run_results.append(metrics)

            logger.info(
                f"Run {run+1} results: CLIP-Score={metrics.get('clip_score', 0):.4f}, "
                f"FID={metrics.get('fid', 0):.2f}, "
                f"DINOv2={metrics.get('dinov2_score', metrics.get('clip_image_score_fallback', 0)):.4f}"
            )

        # Free GPU memory between runs
        del gnn_model, graph_qformer, sd_pipeline, gnn_out
        torch.cuda.empty_cache()

    # ---- Aggregate results ----
    if all_run_results:
        summary = _aggregate_results(all_run_results)
        summary.update({
            "num_runs": num_runs,
            "model": model_name,
            "modality": modality,
            "task": "g2image",
            "hidden_dim": hidden_dim,
            "num_layers": num_layers,
            "lora_rank": lora_rank,
            "resolution": resolution,
            "runs": all_run_results,
        })

        save_path = os.path.join(output_dir, f"g2image_{model_name}_{modality}.json")
        with open(save_path, "w") as f:
            json.dump(summary, f, indent=2, default=str)
        logger.info(f"G2Image results saved to {save_path}")
    else:
        summary = {"error": "No successful runs completed"}
        logger.error("G2Image: no successful runs completed")

    # ---- Neighbor ablation (Table 2) ----
    logger.info("\n" + "=" * 60)
    logger.info("Running neighbor ablation (Table 2)")
    logger.info("=" * 60)

    ablation_results = run_neighbor_ablation_g2image(
        dataset=dataset,
        model_name=model_name,
        modality=modality,
        data_root=data_root,
        paper_ids=paper_ids,
        domain_labels=domain_labels,
        domain_keys=domain_keys,
        neighbor_counts=[0, 2, 4, 6, 8],
        num_runs=num_runs,
        num_epochs=min(num_epochs, 10),  # fewer epochs for ablation
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        dropout=dropout,
        lr=lr,
        batch_size=batch_size,
        device=device,
        output_dir=output_dir,
    )
    summary["neighbor_ablation"] = ablation_results

    # Save combined results
    save_path = os.path.join(output_dir, f"g2image_{model_name}_{modality}.json")
    with open(save_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    logger.info(f"Full G2Image results (incl. ablation) saved to {save_path}")

    return summary


# ---------------------------------------------------------------------------
# 9. Helper utilities
# ---------------------------------------------------------------------------

def _filter_to_papers_with_images(
    node_indices: np.ndarray,
    paper_ids: List[str],
    domain_labels: torch.Tensor,
    domain_keys: List[str],
    data_root: str,
) -> np.ndarray:
    """Keep only node indices whose papers have at least one image file."""
    domain_keys_list = domain_keys
    filtered = []
    for idx in node_indices:
        pid = paper_ids[idx]
        dlabel = domain_labels[idx].item()
        dkey = domain_keys_list[dlabel]
        img_dir = os.path.join(data_root, f"{dkey}_papers", pid, "images")
        if os.path.isdir(img_dir):
            imgs = [f for f in os.listdir(img_dir) if f.lower().endswith((".png", ".jpg", ".jpeg"))]
            if imgs:
                filtered.append(idx)
    return np.array(filtered, dtype=np.int64)


def _load_paper_titles(
    data_root: str,
    paper_ids: List[str],
    domain_labels: torch.Tensor,
    domain_keys: List[str],
) -> Dict[str, str]:
    """
    Load paper titles from markdown files for CLIP-Score evaluation.
    Falls back to paper_id as title if no .md file is found.
    """
    titles = {}
    for i, pid in enumerate(paper_ids):
        dlabel = domain_labels[i].item()
        dkey = domain_keys[dlabel]
        md_path = os.path.join(data_root, f"{dkey}_papers", pid, f"{pid}.md")
        if os.path.exists(md_path):
            try:
                with open(md_path, "r", encoding="utf-8", errors="ignore") as f:
                    first_line = f.readline().strip()
                    # Strip markdown heading markers
                    title = first_line.lstrip("#").strip()
                    if title:
                        titles[pid] = title
                        continue
            except Exception:
                pass
        # Fallback: use paper_id
        titles[pid] = pid
    return titles


def _collect_eval_data(
    node_indices: List[int],
    paper_ids: List[str],
    paper_titles: Dict[str, str],
    domain_labels: torch.Tensor,
    domain_keys: List[str],
    data_root: str,
) -> Tuple[List[Image.Image], List[str]]:
    """Collect real images and text titles for evaluation nodes."""
    real_images = []
    text_titles = []

    for idx in node_indices:
        pid = paper_ids[idx]
        dlabel = domain_labels[idx].item()
        dkey = domain_keys[dlabel]
        img_dir = os.path.join(data_root, f"{dkey}_papers", pid, "images")

        if not os.path.isdir(img_dir):
            continue

        # Use the first image as the representative real image
        img_files = sorted(
            [f for f in os.listdir(img_dir) if f.lower().endswith((".png", ".jpg", ".jpeg"))]
        )
        if img_files:
            img_path = os.path.join(img_dir, img_files[0])
            try:
                img = Image.open(img_path).convert("RGB").resize((256, 256), Image.BILINEAR)
                real_images.append(img)
                text_titles.append(paper_titles.get(pid, pid))
            except Exception:
                continue

    return real_images, text_titles


def _aggregate_results(all_results: List[Dict]) -> Dict:
    """Compute mean and std across multiple runs for each metric."""
    summary = {}
    if not all_results:
        return summary

    metric_keys = [k for k in all_results[0].keys() if k not in ("training_time",)]
    for key in metric_keys:
        values = [r[key] for r in all_results if key in r]
        if values:
            summary[f"{key}_mean"] = float(np.mean(values))
            summary[f"{key}_std"] = float(np.std(values))

    # Also aggregate training time
    times = [r.get("training_time", 0) for r in all_results]
    summary["training_time_mean"] = float(np.mean(times))
    summary["training_time_std"] = float(np.std(times))

    return summary

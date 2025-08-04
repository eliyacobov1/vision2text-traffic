"""Streamlit demo showcasing the vision-language congestion model."""

import argparse
import logging
from io import BytesIO
from pathlib import Path
from typing import Optional, List, Tuple

import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
import torch
import torch.nn.functional as F
import yaml
from PIL import Image
import requests

from model import VisionLanguageTransformer, VLTConfig
from encoders import SimpleTokenizer
from utils import get_transforms, TrafficDataset
from transformers import AutoTokenizer


# -----------------------------------------------------------------------------
# Loading utilities


def load_config(cfg_path: str | None) -> Tuple[VLTConfig, dict]:
    """Load YAML config if present and return the dataclass and raw dict."""

    cfg_dict: dict = {}
    config = VLTConfig()
    if cfg_path and Path(cfg_path).exists():
        with open(cfg_path, "r") as f:
            cfg_dict = yaml.safe_load(f) or {}
        config = VLTConfig(**cfg_dict.get("model", {}))
    return config, cfg_dict


def load_model(ckpt: str | None, cfg_path: str | None, offline: bool = False) -> tuple[VisionLanguageTransformer, dict]:
    """Instantiate model from checkpoint and config with graceful fallbacks."""

    config, cfg_dict = load_config(cfg_path)
    model = VisionLanguageTransformer(config, offline=offline)
    if ckpt and Path(ckpt).exists():
        state = torch.load(ckpt, map_location="cpu")
        model.load_state_dict(state, strict=False)
    else:
        logging.warning("Checkpoint %s not found; using randomly initialised weights", ckpt)
    model.eval()
    return model, cfg_dict


# -----------------------------------------------------------------------------
# Dataset encoding helpers


def encode_dataset_images(model: VisionLanguageTransformer, dataset: TrafficDataset) -> torch.Tensor:
    """Encode dataset images for similarity search."""

    loader = torch.utils.data.DataLoader(dataset, batch_size=4)
    img_embs: List[torch.Tensor] = []
    with torch.no_grad():
        for imgs, _, _, _ in loader:
            img_tokens = model.img_proj(model.vision(imgs))
            if "contrastive" in model.heads:
                head = model.heads["contrastive"]
                emb = F.normalize(head.img_proj(img_tokens.mean(dim=1)), dim=-1)
            else:
                emb = img_tokens.mean(dim=1)
            img_embs.append(emb)
    return torch.cat(img_embs) if img_embs else torch.empty(0, model.config.hidden_dim)


# -----------------------------------------------------------------------------
# Misc helpers


def log_query(text: str, mode: str, log_file: Path):
    with log_file.open("a") as f:
        f.write(f"{mode}\t{text}\n")


def validate_image(uploaded) -> Optional[Image.Image]:
    """Ensure uploaded file is an image and not overly large."""

    if uploaded.type not in {"image/jpeg", "image/png", "image/jpg"}:
        st.error("Only JPEG or PNG images are supported")
        return None
    if uploaded.size and uploaded.size > 5 * 1024 * 1024:
        st.error("Image file too large (>5MB)")
        return None
    try:
        return Image.open(uploaded).convert("RGB")
    except Exception:
        st.error("Invalid image file")
        return None


def compute_heatmap(attn: torch.Tensor, image: Image.Image):
    """Convert an attention tensor to a matplotlib figure overlay.

    The attention tensor may be either ``(B, heads, Q, K)`` when per-head
    weights are returned or ``(B, Q, K)`` if already averaged. In both cases we
    select the attention weights for the first query token and reshape them
    into a square grid matching the vision backbone's patch layout.
    """

    if attn.ndim == 4:
        attn_map = attn[0].mean(0)[0]
    elif attn.ndim == 3:
        attn_map = attn[0][0]
    else:
        raise ValueError(f"Unexpected attention shape: {attn.shape}")

    # Drop the [CLS] token if present so the remaining tokens form a square grid
    if attn_map.numel() > 1:
        attn_map = attn_map[1:]
    num_tokens = attn_map.numel()
    grid = int(num_tokens ** 0.5)
    heat = attn_map.reshape(grid, grid).cpu().numpy()
    heat = (heat - heat.min()) / (heat.max() - heat.min() + 1e-6)
    heat = np.kron(heat, np.ones((image.size[1] // grid, image.size[0] // grid)))
    fig, ax = plt.subplots()
    ax.imshow(image)
    ax.imshow(heat, cmap="jet", alpha=0.5, extent=(0, image.size[0], image.size[1], 0))
    ax.axis("off")
    return fig


def is_frozen(module: torch.nn.Module) -> bool:
    return not any(p.requires_grad for p in module.parameters())


def show_model_info(model: VisionLanguageTransformer, cfg_dict: dict):
    """Display model architecture summary and config."""

    with st.expander("Model information", expanded=False):
        st.write(
            f"Vision encoder: {model.config.vision_model} "
            f"({'frozen' if is_frozen(model.vision) else 'trainable'})"
        )
        st.write(
            f"Text encoder: {model.config.text_model} "
            f"({'frozen' if is_frozen(model.text) else 'trainable'})"
        )
        st.write(f"Cross-attention layers: {model.config.num_layers}")
        st.write(f"Hidden dimension: {model.config.hidden_dim}")
        total = sum(p.numel() for p in model.parameters())
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        st.write(f"Parameters: {total:,} (trainable {trainable:,})")
        if cfg_dict:
            st.write("YAML config values:")
            st.json(cfg_dict)


# -----------------------------------------------------------------------------
# Streamlit app


def run_app(args: Optional[argparse.Namespace] = None):
    st.set_page_config(page_title="Traffic Congestion Detector", layout="wide")
    st.title("Traffic Congestion Detector")
    st.markdown(
        "Upload a road image or pick a sample and query the model with a custom prompt "
        "to assess congestion levels."
    )

    # Sidebar / advanced options
    with st.sidebar.expander("Advanced options", expanded=False):
        ckpt = st.text_input("Checkpoint path", args.ckpt if args else "checkpoints/model.pt")
        config_path = st.text_input("Config path", getattr(args, "config", "config.yaml") if args else "config.yaml")
        offline = st.checkbox("Offline mode", value=False)
        data_dir = st.text_input("Data dir for retrieval", args.data_dir if args else "sample_data")

    head = st.sidebar.radio("Model head", ["classification", "contrastive"])
    templates_str = st.sidebar.text_area(
        "Prompt templates (one per line)", "Describe traffic\nTraffic density:"
    )
    templates = [t.strip() for t in templates_str.splitlines() if t.strip()]

    model, cfg_dict = load_model(ckpt, config_path, offline)
    if ckpt and not Path(ckpt).exists():
        st.warning(f"Checkpoint '{ckpt}' not found. Using random weights.")
    if config_path and not Path(config_path).exists():
        st.warning(f"Config '{config_path}' not found. Using defaults.")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model.config.text_model, local_files_only=offline)
    except Exception:
        tokenizer = SimpleTokenizer()
    transform = get_transforms()

    dataset = TrafficDataset(data_dir, model.config.text_model, offline=offline)
    retrieval_img_embs = encode_dataset_images(model, dataset) if head == "contrastive" else None

    sample_names = [f"{i}: {t}" for i, (_, t, _) in enumerate(dataset.samples)]
    sample_choice = st.sidebar.selectbox("Sample gallery", ["None"] + sample_names)

    uploaded = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])
    image: Optional[Image.Image] = None
    if uploaded is not None:
        image = validate_image(uploaded)
    elif sample_choice != "None":
        url, _, _ = dataset.samples[int(sample_choice.split(":")[0])]
        try:
            headers = {"User-Agent": "Mozilla/5.0"}
            resp = requests.get(url, timeout=10, headers=headers)
            image = Image.open(BytesIO(resp.content)).convert("RGB")
        except Exception as e:  # pragma: no cover - network failure is unlikely
            st.error(f"Failed to load sample image: {e}")

    prompt = st.text_input("Enter a prompt", "Is this road congested?")

    if image is not None and prompt:
        st.image(image, caption="Input image", use_column_width=True)
        img_tensor = transform(image).unsqueeze(0)
        prompts = [tpl.replace("{}", prompt) if "{}" in tpl else f"{tpl} {prompt}" for tpl in templates] or [prompt]
        probs: List[float] = []
        attn = None
        img_emb = txt_emb = None
        for p in prompts:
            tokens = tokenizer(p, return_tensors="pt", padding=True)
            with torch.no_grad():
                out = model(img_tensor, tokens["input_ids"], tokens["attention_mask"])
                attn = model.get_last_attention()
            if isinstance(out, dict):
                prob = out.get("classification")
                img_emb, txt_emb = out.get("contrastive", (None, None))
            else:
                prob, img_emb, txt_emb = out, None, None
            probs.append(float(prob))
        avg_prob = sum(probs) / len(probs)
        logit = float(np.log(avg_prob / (1 - avg_prob + 1e-6)))
        st.metric("Congestion probability", f"{avg_prob:.3f}", help=f"logit {logit:.3f}")
        st.progress(min(max(avg_prob, 0.0), 1.0))

        if head == "contrastive" and img_emb is not None and retrieval_img_embs is not None and len(retrieval_img_embs) > 0:
            sims = (retrieval_img_embs @ img_emb.T).squeeze(1)
            top_idx = int(torch.topk(sims, 1).indices[0])
            url, caption, _ = dataset.samples[top_idx]
            headers = {"User-Agent": "Mozilla/5.0"}
            resp = requests.get(url, timeout=10, headers=headers)
            top_img = Image.open(BytesIO(resp.content)).convert("RGB")
            st.write("Top matching training image:")
            st.image(top_img, caption=caption, use_column_width=True)

        if attn is not None:
            fig = compute_heatmap(attn, image)
            st.pyplot(fig)

        log_file = Path("logs/demo_queries.log")
        log_file.parent.mkdir(exist_ok=True)
        log_query(prompt, head, log_file)

    show_model_info(model, cfg_dict)


# -----------------------------------------------------------------------------
# Main


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", default="checkpoints/model.pt")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--data-dir", default="sample_data")
    args = parser.parse_args()
    run_app(args)


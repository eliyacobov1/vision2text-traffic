import argparse
import logging
from pathlib import Path
from typing import Optional

import streamlit as st
from PIL import Image
import numpy as np
import torch
import matplotlib.pyplot as plt

from model import VisionLanguageTransformer, VLTConfig, SimpleTokenizer
from utils import get_transforms
from transformers import AutoTokenizer


def load_model(ckpt: str, offline: bool = False) -> VisionLanguageTransformer:
    config = VLTConfig()
    model = VisionLanguageTransformer(config, offline=offline)
    if Path(ckpt).exists():
        model.load_state_dict(torch.load(ckpt, map_location="cpu"))
    model.eval()
    return model


def log_query(text: str, mode: str, log_file: Path):
    with log_file.open("a") as f:
        f.write(f"{mode}\t{text}\n")


def run_app(args: Optional[argparse.Namespace] = None):
    st.title("Traffic Congestion Detector")
    ckpt = st.sidebar.text_input("Checkpoint path", args.ckpt if args else "checkpoints/model.pt")
    offline = st.sidebar.checkbox("Offline mode", value=False)
    mode = st.sidebar.selectbox("Mode", ["classification", "contrastive"])  # choose output

    model = load_model(ckpt, offline=offline)
    try:
        tokenizer = AutoTokenizer.from_pretrained(model.config.text_model, local_files_only=offline)
    except Exception:
        tokenizer = SimpleTokenizer()
    transform = get_transforms()

    uploaded = st.file_uploader("Upload an image")
    text = st.text_input("Describe the traffic level in this scene.")

    if uploaded and text:
        image = Image.open(uploaded).convert("RGB")
        st.image(image, caption="Input Image")
        img_tensor = transform(image).unsqueeze(0)
        tokens = tokenizer(text, return_tensors="pt", padding=True)
        with torch.no_grad():
            out = model(img_tensor, tokens["input_ids"], tokens["attention_mask"])
            attn = model.get_last_attention()
        if isinstance(out, tuple):
            prob, img_emb, txt_emb = out
        else:
            prob, img_emb, txt_emb = out, None, None

        if mode == "classification" or prob is not None:
            st.write(f"Congestion probability: {float(prob):.3f}")

        if mode == "contrastive" and img_emb is not None and txt_emb is not None:
            sim = float((img_emb @ txt_emb.T).item())
            st.write(f"Similarity score: {sim:.3f}")

        if attn is not None:
            attn_map = attn[0].mean(0)[0]
            grid = int(len(attn_map) ** 0.5)
            heat = attn_map.reshape(grid, grid).cpu().numpy()
            heat = (heat - heat.min()) / (heat.max() - heat.min() + 1e-6)
            heat = np.kron(heat, np.ones((16, 16)))
            plt.imshow(image)
            plt.imshow(heat, cmap="jet", alpha=0.5)
            st.pyplot(plt)

        log_file = Path("logs/demo_queries.log")
        log_file.parent.mkdir(exist_ok=True)
        log_query(text, mode, log_file)


if __name__ == "__main__":
    run_app()

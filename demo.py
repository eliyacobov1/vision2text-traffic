import argparse
import logging
from pathlib import Path
from typing import Optional, List

import streamlit as st
from PIL import Image
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

from model import VisionLanguageTransformer, VLTConfig
from encoders import SimpleTokenizer
from utils import get_transforms, TrafficDataset
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


def encode_dataset_texts(model: VisionLanguageTransformer, dataset: TrafficDataset) -> torch.Tensor:
    texts = [t for _, t, _ in dataset.samples]
    ids, masks = model.encode_text(texts)
    with torch.no_grad():
        txt_tokens = model.txt_proj(model.text(ids, masks))
        cls = txt_tokens[:, 0, :]
        if "contrastive" in model.heads:
            head = model.heads["contrastive"]
            txt_emb = torch.nn.functional.normalize(head.txt_proj(cls), dim=-1)
        else:
            txt_emb = cls
    return txt_emb


def run_app(args: Optional[argparse.Namespace] = None):
    st.title("Traffic Congestion Detector")
    ckpt = st.sidebar.text_input("Checkpoint path", args.ckpt if args else "checkpoints/model.pt")
    offline = st.sidebar.checkbox("Offline mode", value=False)
    data_dir = st.sidebar.text_input("Data dir for retrieval", args.data_dir if args else "sample_data")
    mode = st.sidebar.selectbox("Mode", ["classification", "contrastive"])  # choose output
    templates_str = st.sidebar.text_area("Prompt templates (one per line)", "Describe traffic\nTraffic density:")
    templates = [t.strip() for t in templates_str.splitlines() if t.strip()]

    model = load_model(ckpt, offline=offline)
    try:
        tokenizer = AutoTokenizer.from_pretrained(model.config.text_model, local_files_only=offline)
    except Exception:
        tokenizer = SimpleTokenizer()
    transform = get_transforms()

    dataset = TrafficDataset(data_dir, model.config.text_model, offline=offline)
    retrieval_embs = encode_dataset_texts(model, dataset) if mode == "contrastive" else None
    retrieval_texts = [t for _, t, _ in dataset.samples]

    uploaded = st.file_uploader("Upload an image")
    text = st.text_input("Describe the traffic level in this scene.")

    if uploaded and text:
        image = Image.open(uploaded).convert("RGB")
        st.image(image, caption="Input Image")
        img_tensor = transform(image).unsqueeze(0)
        prompts = [tpl.replace("{}", text) if "{}" in tpl else f"{tpl} {text}" for tpl in templates] or [text]
        probs = []
        attn = None
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
        st.write(f"Congestion probability: {avg_prob:.3f}")

        if mode == "contrastive" and img_emb is not None and txt_emb is not None:
            sim = float((img_emb @ txt_emb.T).item())
            st.write(f"Similarity score: {sim:.3f}")
            if retrieval_embs is not None:
                sims = (retrieval_embs @ txt_emb.T).squeeze(1)
                topk = torch.topk(sims, min(3, len(retrieval_texts))).indices.tolist()
                st.write("Top similar prompts:")
                for idx in topk:
                    st.write(f"{retrieval_texts[idx]} (score {sims[idx]:.3f})")

        if attn is not None:
            attn_map = attn[0].mean(0)[0]
            grid = int(len(attn_map) ** 0.5)
            heat = attn_map.reshape(grid, grid).cpu().numpy()
            heat = (heat - heat.min()) / (heat.max() - heat.min() + 1e-6)
            heat = np.kron(heat, np.ones((image.size[0] // grid, image.size[1] // grid)))
            plt.imshow(image)
            plt.imshow(heat, cmap="jet", alpha=0.5)
            st.pyplot(plt)

        log_file = Path("logs/demo_queries.log")
        log_file.parent.mkdir(exist_ok=True)
        log_query(text, mode, log_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", default="checkpoints/model.pt")
    parser.add_argument("--data-dir", default="sample_data")
    args = parser.parse_args()
    run_app(args)

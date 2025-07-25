import streamlit as st
from PIL import Image
from pathlib import Path
import torch
from model import VisionLanguageTransformer, VLTConfig, SimpleTokenizer
from utils import get_transforms
from transformers import AutoTokenizer


def load_model(ckpt: str, offline: bool = False):
    config = VLTConfig()
    model = VisionLanguageTransformer(config)
    if Path(ckpt).exists():
        model.load_state_dict(torch.load(ckpt, map_location='cpu'))
    model.eval()
    return model


st.title('Traffic Congestion Detector')
ckpt = st.sidebar.text_input('Checkpoint path', 'checkpoints/model.pt')
offline = st.sidebar.checkbox('Offline mode', value=False)
model = load_model(ckpt, offline=offline)
try:
    tokenizer = AutoTokenizer.from_pretrained(model.config.text_model, local_files_only=offline)
except Exception:
    tokenizer = SimpleTokenizer()
transform = get_transforms()

uploaded = st.file_uploader('Upload an image')
text = st.text_input('Describe the traffic level in this scene.')

if uploaded and text:
    image = Image.open(uploaded).convert('RGB')
    st.image(image, caption='Input Image')
    img_tensor = transform(image).unsqueeze(0)
    tokens = tokenizer(text, return_tensors='pt', padding=True)
    with torch.no_grad():
        pred = model(img_tensor, tokens['input_ids'], tokens['attention_mask'])
    st.write(f'Congestion probability: {pred.item():.3f}')

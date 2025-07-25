import csv
from io import BytesIO
from pathlib import Path
from typing import List, Tuple

import requests

import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
from transformers import AutoTokenizer
from encoders import SimpleTokenizer


def get_transforms(image_size: int = 224):
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
    ])


class TrafficDataset(Dataset):
    """Dataset loading (image_url, text, label) triplets.

    If ``root`` contains a ``dataset.csv`` file, samples are read from it. The
    CSV must have ``image_url``, ``text`` and ``label`` columns. Images are
    downloaded from the URLs at runtime. Otherwise a legacy directory structure
    with ``images/``, ``texts/`` and ``labels/`` folders is supported.
    """

    def __init__(self, root: str, tokenizer_name: str, image_size: int = 224, offline: bool = False):
        self.root = Path(root)
        self.samples = []
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, local_files_only=offline)
        except Exception:
            self.tokenizer = SimpleTokenizer()
        self.tfms = get_transforms(image_size)

        csv_file = self.root / "dataset.csv"
        if csv_file.exists():
            with open(csv_file, newline="") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    url = row.get("image_url") or row.get("image")
                    text = row.get("text", "")
                    label = int(row.get("label", 0))
                    self.samples.append((url, text, label))
        else:
            for img in sorted((self.root / "images").glob("*.jpg")):
                label_file = self.root / "labels" / f"{img.stem}.txt"
                text_file = self.root / "texts" / f"{img.stem}.txt"
                if label_file.exists() and text_file.exists():
                    label = int(label_file.read_text().strip())
                    text = text_file.read_text().strip()
                    self.samples.append((img.as_posix(), text, label))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
        img_src, text, label = self.samples[idx]
        if str(img_src).startswith("http"):
            headers = {"User-Agent": "Mozilla/5.0"}
            resp = requests.get(img_src, timeout=10, headers=headers)
            image = Image.open(BytesIO(resp.content)).convert("RGB")
        else:
            image = Image.open(img_src).convert("RGB")
        image = self.tfms(image)
        tokens = self.tokenizer(text, return_tensors="pt", padding="max_length", truncation=True, max_length=32)
        input_ids = tokens["input_ids"].squeeze(0)
        attention_mask = tokens["attention_mask"].squeeze(0)
        return image, input_ids, attention_mask, label

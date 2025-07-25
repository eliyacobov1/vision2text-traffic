import argparse
import logging
from pathlib import Path
import yaml

import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.nn.functional import binary_cross_entropy
from torch.cuda.amp import GradScaler, autocast
from sklearn.metrics import f1_score

from model import VisionLanguageTransformer, VLTConfig
from utils import TrafficDataset


def load_config(path: str):
    with open(path) as f:
        cfg = yaml.safe_load(f)
    model_cfg = cfg.get("model", {})
    train_cfg = cfg.get("training", {})
    data_cfg = cfg.get("data", {})
    return VLTConfig(**model_cfg), train_cfg, data_cfg


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    config, train_cfg, data_cfg = load_config(args.config)
    model = VisionLanguageTransformer(config, offline=args.offline).to(device)

    dataset = TrafficDataset(data_cfg.get("root", args.data_dir), config.text_model, offline=args.offline)
    loader = DataLoader(dataset, batch_size=train_cfg.get("batch_size", args.batch_size), shuffle=True)

    optimizer = Adam(model.parameters(), lr=train_cfg.get("lr", args.lr))
    scaler = GradScaler(enabled=train_cfg.get("amp", False))

    patience = train_cfg.get("early_stop_patience", 3)
    best_f1 = 0.0
    patience_ctr = 0

    logging.info("Starting training for %d epochs", train_cfg.get("epochs", args.epochs))
    for epoch in range(train_cfg.get("epochs", args.epochs)):
        model.train()
        total_loss = 0.0
        all_preds = []
        all_labels = []
        for images, input_ids, attention_mask, labels in loader:
            images = images.to(device)
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            labels = labels.float().to(device)

            with autocast(enabled=train_cfg.get("amp", False)):
                out = model(images, input_ids, attention_mask)
                preds = out[0] if isinstance(out, tuple) else out
                loss = binary_cross_entropy(preds, labels)

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item() * len(labels)
            all_preds.extend(preds.detach().cpu())
            all_labels.extend(labels.cpu())

        avg_loss = total_loss / len(dataset)
        f1 = f1_score(torch.tensor(all_labels), torch.tensor(all_preds) > 0.5)
        logging.info("Epoch %d: loss=%.4f f1=%.4f", epoch + 1, avg_loss, f1)

        if f1 > best_f1:
            best_f1 = f1
            patience_ctr = 0
            Path(train_cfg.get("checkpoint_dir", args.out_dir)).mkdir(parents=True, exist_ok=True)
            ckpt = Path(train_cfg.get("checkpoint_dir", args.out_dir)) / "model.pt"
            torch.save(model.state_dict(), ckpt)
            logging.info("Saved checkpoint to %s", ckpt)
        else:
            patience_ctr += 1
            if patience_ctr >= patience:
                logging.info("Early stopping")
                break

    Path(train_cfg.get("checkpoint_dir", args.out_dir)).mkdir(parents=True, exist_ok=True)
    ckpt = Path(train_cfg.get("checkpoint_dir", args.out_dir)) / "model_last.pt"
    torch.save(model.state_dict(), ckpt)
    logging.info("Final checkpoint saved to %s", ckpt)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--config', default='config.yaml', help='Path to config YAML')
    p.add_argument('--data-dir', default='sample_data')
    p.add_argument('--out-dir', default='checkpoints')
    p.add_argument('--epochs', type=int, default=5)
    p.add_argument('--batch-size', type=int, default=4)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--offline', action='store_true', help='Use local files only')
    return p.parse_args()


if __name__ == '__main__':
    args = parse_args()
    train(args)

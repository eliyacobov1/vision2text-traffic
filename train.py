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
    ckpt_dir = Path(train_cfg.get("checkpoint_dir", args.checkpoint_dir))
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    model = VisionLanguageTransformer(config, offline=args.offline).to(device)

    dataset = TrafficDataset(data_cfg.get("root", args.data_dir), config.text_model, offline=args.offline)
    batch_size = int(train_cfg.get("batch_size", args.batch_size))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    lr = float(train_cfg.get("lr", args.lr))
    optimizer = Adam(model.parameters(), lr=lr)
    scaler = GradScaler(enabled=train_cfg.get("amp", False))

    patience = train_cfg.get("early_stop_patience", 3)
    best_f1 = 0.0
    patience_ctr = 0
    history = []

    epochs = int(train_cfg.get("epochs", args.epochs))
    logging.info("Starting training for %d epochs", epochs)
    for epoch in range(epochs):
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
                preds = out["classification"] if isinstance(out, dict) else out
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
        history.append({"epoch": epoch + 1, "loss": avg_loss, "f1": float(f1)})
        logging.info("Epoch %d: loss=%.4f f1=%.4f", epoch + 1, avg_loss, f1)

        if f1 > best_f1:
            best_f1 = f1
            patience_ctr = 0
            ckpt = ckpt_dir / "model.pt"
            torch.save(model.state_dict(), ckpt)
            logging.info("Saved checkpoint to %s", ckpt)
        else:
            patience_ctr += 1
            if patience_ctr >= patience:
                logging.info("Early stopping")
                break

    ckpt = ckpt_dir / "model_last.pt"
    torch.save(model.state_dict(), ckpt)
    logging.info("Final checkpoint saved to %s", ckpt)

    # save metrics
    import json
    metrics_path = ckpt_dir / "metrics.json"
    with metrics_path.open("w") as f:
        json.dump({"history": history}, f, indent=2)

    import pandas as pd
    df = pd.DataFrame(history)
    df.to_csv(ckpt_dir / "metrics.csv", index=False)

    # plot loss/f1 curves
    import matplotlib.pyplot as plt
    epochs = [h["epoch"] for h in history]
    losses = [h["loss"] for h in history]
    f1s = [h["f1"] for h in history]
    plt.figure()
    plt.plot(epochs, losses, label="loss")
    plt.plot(epochs, f1s, label="f1")
    plt.xlabel("Epoch")
    plt.legend()
    plt.title("Training curves")
    plt.savefig(ckpt_dir / "train_curves.png")
    plt.close()


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--config', default='config.yaml', help='Path to config YAML')
    p.add_argument('--data-dir', default='sample_data')
    p.add_argument('--checkpoint-dir', default='checkpoints')
    p.add_argument('--epochs', type=int, default=5)
    p.add_argument('--batch-size', type=int, default=4)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--offline', action='store_true', help='Use local files only')
    return p.parse_args()


if __name__ == '__main__':
    args = parse_args()
    train(args)

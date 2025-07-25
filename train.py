import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.nn.functional import binary_cross_entropy

from model import VisionLanguageTransformer, VLTConfig
from utils import TrafficDataset


def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    config = VLTConfig()
    model = VisionLanguageTransformer(config).to(device)

    dataset = TrafficDataset(args.data_dir, config.text_model, offline=args.offline)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    optimizer = Adam(model.parameters(), lr=args.lr)

    best_loss = float('inf')
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        correct = 0
        for images, input_ids, attention_mask, labels in loader:
            images = images.to(device)
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            labels = labels.float().to(device)

            preds = model(images, input_ids, attention_mask)
            loss = binary_cross_entropy(preds, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * len(labels)
            correct += ((preds > 0.5).long() == labels.long()).sum().item()

        avg_loss = total_loss / len(dataset)
        acc = correct / len(dataset)
        print(f"Epoch {epoch+1}: loss={avg_loss:.4f} acc={acc:.4f}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            Path(args.out_dir).mkdir(parents=True, exist_ok=True)
            ckpt = Path(args.out_dir) / 'model.pt'
            torch.save(model.state_dict(), ckpt)
            print(f"Saved checkpoint to {ckpt}")

    # Always save final checkpoint for convenience
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    ckpt = Path(args.out_dir) / 'model_last.pt'
    torch.save(model.state_dict(), ckpt)
    print(f"Final checkpoint saved to {ckpt}")


def parse_args():
    p = argparse.ArgumentParser()
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

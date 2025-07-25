import argparse
import logging
from pathlib import Path
import yaml
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score, roc_curve, precision_recall_curve, confusion_matrix
from torch.utils.data import DataLoader

from model import VisionLanguageTransformer, VLTConfig
from utils import TrafficDataset


def load_config(path: str):
    with open(path) as f:
        cfg = yaml.safe_load(f)
    model_cfg = cfg.get("model", {})
    train_cfg = cfg.get("training", {})
    data_cfg = cfg.get("data", {})
    return VLTConfig(**model_cfg), train_cfg, data_cfg


def evaluate(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config, train_cfg, data_cfg = load_config(args.config)
    model = VisionLanguageTransformer(config, offline=args.offline).to(device)
    model.load_state_dict(torch.load(args.ckpt, map_location=device))
    model.eval()
    logging.info("Loaded model from %s", args.ckpt)

    dataset = TrafficDataset(data_cfg.get("root", args.data_dir), config.text_model, offline=args.offline)
    loader = DataLoader(dataset, batch_size=args.batch_size)

    preds = []
    labels = []
    with torch.no_grad():
        for images, input_ids, attention_mask, lbls in loader:
            images = images.to(device)
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            outputs = model(images, input_ids, attention_mask)
            preds.extend(outputs.cpu().tolist())
            labels.extend(lbls.tolist())

    preds_tensor = torch.tensor(preds)
    labels_tensor = torch.tensor(labels).float()
    precision, recall, f1, _ = precision_recall_fscore_support(labels_tensor, preds_tensor > 0.5, average="binary")
    auc = roc_auc_score(labels_tensor, preds_tensor)
    logging.info("Precision: %.4f Recall: %.4f F1: %.4f AUC: %.4f", precision, recall, f1, auc)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fpr, tpr, _ = roc_curve(labels_tensor, preds_tensor)
    plt.figure()
    plt.plot(fpr, tpr)
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.title("ROC")
    plt.savefig(out_dir / "roc.png")
    plt.close()

    prec, rec, _ = precision_recall_curve(labels_tensor, preds_tensor)
    plt.figure()
    plt.plot(rec, prec)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("PR Curve")
    plt.savefig(out_dir / "pr_curve.png")
    plt.close()

    cm = confusion_matrix(labels_tensor, preds_tensor > 0.5)
    plt.figure()
    plt.imshow(cm, cmap="Blues")
    plt.title("Confusion Matrix")
    plt.savefig(out_dir / "confusion_matrix.png")
    plt.close()

    for i in range(min(5, len(dataset))):
        img_src, text, label = dataset.samples[i]
        logging.info("Image: %s | Text: %s | Label: %s | Pred: %.3f", img_src, text, label, preds[i])

    logging.info("Evaluation complete. Plots saved to %s", out_dir)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--config', default='config.yaml', help='Path to config YAML')
    p.add_argument('--data-dir', default='sample_data')
    p.add_argument('--ckpt', default='checkpoints/model.pt')
    p.add_argument('--batch-size', type=int, default=4)
    p.add_argument('--offline', action='store_true', help='Use local files only')
    p.add_argument('--out-dir', default='eval_outputs')
    return p.parse_args()


if __name__ == '__main__':
    args = parse_args()
    evaluate(args)

import argparse
import torch
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score
from torch.utils.data import DataLoader

from model import VisionLanguageTransformer, VLTConfig
from utils import TrafficDataset


def evaluate(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    config = VLTConfig()
    model = VisionLanguageTransformer(config).to(device)
    model.load_state_dict(torch.load(args.ckpt, map_location=device))
    model.eval()

    dataset = TrafficDataset(args.data_dir, config.text_model, offline=args.offline)
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
    precision, recall, f1, _ = precision_recall_fscore_support(labels_tensor, preds_tensor > 0.5, average='binary')
    auc = roc_auc_score(labels_tensor, preds_tensor)
    print(f"Precision: {precision:.4f} Recall: {recall:.4f} F1: {f1:.4f} AUC: {auc:.4f}")

    for i in range(min(5, len(dataset))):
        img_src, text, label = dataset.samples[i]
        print(f"Image: {img_src} | Text: {text} | Label: {label} | Pred: {preds[i]:.3f}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--data-dir', default='sample_data')
    p.add_argument('--ckpt', default='checkpoints/model.pt')
    p.add_argument('--batch-size', type=int, default=4)
    p.add_argument('--offline', action='store_true', help='Use local files only')
    return p.parse_args()


if __name__ == '__main__':
    args = parse_args()
    evaluate(args)

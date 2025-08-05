import argparse


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run VLT experiment")
    parser.add_argument("--config", default="config.yaml", help="path to config")
    parser.add_argument("--ckpt", default="checkpoints/model.pt", help="checkpoint for eval/demo")
    parser.add_argument("--offline", action="store_true", help="offline mode")
    parser.add_argument("--data-dir", default="sample_data", help="dataset directory")
    parser.add_argument("--out-dir", default="eval_outputs", help="eval output directory for eval mode")
    parser.add_argument("--log-dir", default="logs", help="where to save logs")
    parser.add_argument("--checkpoint-dir", default="checkpoints", help="where to save training checkpoints")
    parser.add_argument("--epochs", type=int, default=5, help="number of training epochs")
    parser.add_argument("--batch-size", type=int, default=4, help="mini-batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="learning rate")
    parser.add_argument("--hf-dataset", default=None, help="HuggingFace dataset name")
    parser.add_argument("--hf-split", default="train", help="dataset split to use")
    parser.add_argument("--limit", type=int, default=None, help="limit number of samples")
    return parser


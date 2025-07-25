import argparse
import logging
from pathlib import Path
import yaml

from train import train
from evaluate import evaluate
from demo import run_app


def setup_logging(mode: str, log_dir: str = "logs"):
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    logfile = Path(log_dir) / f"{mode}_log.txt"
    logging.basicConfig(
        filename=logfile,
        level=logging.INFO,
        format="%(asctime)s %(levelname)s: %(message)s",
    )
    logging.getLogger().addHandler(logging.StreamHandler())


def load_config(path: str):
    with open(path) as f:
        cfg = yaml.safe_load(f)
    return cfg


def main():
    parser = argparse.ArgumentParser(description="Run VLT experiment")
    parser.add_argument("mode", choices=["train", "eval", "demo"], help="stage to run")
    parser.add_argument("--config", default="config.yaml", help="path to config")
    parser.add_argument("--ckpt", default="checkpoints/model.pt", help="checkpoint for eval/demo")
    parser.add_argument("--offline", action="store_true", help="offline mode")
    parser.add_argument("--data-dir", default="sample_data", help="dataset directory")
    parser.add_argument("--out-dir", default="eval_outputs", help="eval output directory")
    parser.add_argument("--log-dir", default="logs", help="where to save logs")
    args = parser.parse_args()

    setup_logging(args.mode, args.log_dir)
    logging.info("Running mode %s with config %s", args.mode, args.config)

    if args.mode == "train":
        train(args)
    elif args.mode == "eval":
        evaluate(args)
    else:
        run_app(args)


if __name__ == "__main__":
    main()

import logging
from pathlib import Path
import subprocess
import yaml

from train import train
from evaluate import evaluate
from cli import get_parser


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
    parser = get_parser()
    parser.add_argument("mode", choices=["train", "eval", "demo"], help="stage to run")
    args = parser.parse_args()

    setup_logging(args.mode, args.log_dir)
    logging.info("Running mode %s with config %s", args.mode, args.config)

    if args.mode == "train":
        train(args)
    elif args.mode == "eval":
        evaluate(args)
    else:
        cmd = [
            "streamlit",
            "run",
            str(Path(__file__).parent / "demo.py"),
            "--",
            "--ckpt",
            args.ckpt,
            "--data-dir",
            args.data_dir,
        ]
        if args.offline:
            cmd.append("--offline")
        subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()

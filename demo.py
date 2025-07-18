import argparse
import os
import tempfile
import urllib.request

import main
import cv2

try:
    from tqdm import tqdm
except Exception:  # pragma: no cover - optional progress bar
    tqdm = None

SAMPLE_URL = (
    "https://raw.githubusercontent.com/opencv/opencv/master/samples/data/Megamind.avi"
)


def download_sample(path: str) -> str:
    if not os.path.exists(path):
        print("Downloading sample video...")
        urllib.request.urlretrieve(SAMPLE_URL, path)
    return path


def run_demo(args: argparse.Namespace) -> None:
    if args.video is None:
        tmp_dir = tempfile.gettempdir()
        args.video = os.path.join(tmp_dir, "sample.avi")
        download_sample(args.video)

    main_args = argparse.Namespace(
        input=args.video,
        output=args.output,
        log=args.log,
        model="yolov5s.pt",
        device=args.device,
        caption=args.caption,
        flamingo=args.flamingo,
        simple=args.mode == "simple",
        scratch=args.mode == "scratch",
        scene=getattr(args, "scene", True),
    )

    total = None
    if tqdm and os.path.exists(args.video):
        cap = cv2.VideoCapture(args.video)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or None
        cap.release()

    if tqdm and total:
        print("Processing video...")
        with tqdm(total=total) as pbar:
            def progress_callback():
                pbar.update(1)

            main.main(main_args, progress=progress_callback)  # type: ignore[arg-type]
    else:
        main.main(main_args)

    print(f"Demo complete. Output saved to {args.output}")

    if os.path.exists(args.log):
        lines = open(args.log).read().splitlines()
        congested = sum("Congested" in l for l in lines)
        print(f"Frames processed: {len(lines)}  Congested frames: {congested}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Vision2Text demo")
    parser.add_argument(
        "--mode",
        choices=["full", "simple", "scratch"],
        default="full",
        help="Pipeline configuration",
    )
    parser.add_argument("--video", help="Optional path to input video")
    parser.add_argument("--output", default="demo_output.mp4", help="Output video path")
    parser.add_argument("--log", default="demo_log.txt", help="Log file path")
    parser.add_argument("--device", default="cpu", help="Computation device")
    parser.add_argument("--caption", action="store_true", help="Enable caption generation")
    parser.add_argument("--flamingo", action="store_true", help="Use Flamingo-inspired mode")
    parser.add_argument("--scene", action="store_true", help="Enable unified scene model")
    run_demo(parser.parse_args())

"""Main pipeline for traffic congestion detection and captioning."""

import argparse
from pipeline import PipelineConfig, VisionLanguagePipeline
from detector.yolo_detector import YOLODetector  # for monkeypatching in tests
from captioner.clip_captioner import CLIPCaptioner  # for monkeypatching in tests

# Compatibility alias for older tests
CaptionGenerator = CLIPCaptioner


def run_pipeline(config: PipelineConfig, progress=None) -> None:
    """Wrapper invoking :class:`VisionLanguagePipeline`."""
    pipeline = VisionLanguagePipeline(config)
    pipeline.run(progress=progress)




def main(args: argparse.Namespace, progress=None) -> None:
    """Entry point converting arguments to ``PipelineConfig``."""
    caption_enabled = getattr(args, "caption", None)
    if caption_enabled is None:
        caption_enabled = not getattr(args, "no_caption", False)
    config = PipelineConfig(
        input=args.input,
        output=args.output,
        log=args.log,
        model=args.model,
        caption_model=args.caption_model,
        device=args.device,
        caption=caption_enabled,
    )
    run_pipeline(config, progress=progress)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Traffic congestion detection")
    parser.add_argument("--input", required=True, help="Input video path")
    parser.add_argument("--output", default="output.mp4", help="Output video path")
    parser.add_argument("--log", default="congestion_log.txt", help="Log file path")
    parser.add_argument("--model", default="yolov5s.pt", help="YOLO model weights")
    parser.add_argument(
        "--caption-model",
        default="",
        help="Path to a fine-tuned CLIP model",
    )
    parser.add_argument("--device", default="cpu", help="Computation device")
    parser.add_argument(
        "--no-caption",
        action="store_true",
        help="Skip caption generation (testing only)",
    )
    main(parser.parse_args())


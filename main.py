"""Main pipeline for traffic congestion detection and captioning."""

import argparse
from typing import List, Dict

import cv2

from detector.yolo_detector import YOLODetector
from detector.simple_detector import SimpleMotionDetector
from detector.cnn_detector import CNNDetector
from analyzer.congestion_detector import CongestionDetector
from traffic_utils.video import create_writer
from flamingo.vision_text_model import FlamingoVisionTextModel
from scene.scene_model import SceneUnderstandingModel

try:
    from captioner.generate_caption import CaptionGenerator
    from captioner.simple_captioner import SimpleCaptioner
    from captioner.transformer_captioner import VisionLanguageModel
except Exception:  # pragma: no cover - captioning is optional
    CaptionGenerator = None  # type: ignore


def annotate_frame(frame, detections: List[Dict], status: str, caption: str = ""):
    for det in detections:
        x1, y1, x2, y2 = map(int, det["bbox"])
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"{det['label']} {det['conf']:.2f}", (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    cv2.putText(frame, status, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1,
                (0, 0, 255) if status == "Congested" else (0, 255, 0), 2)

    if caption:
        cv2.putText(frame, caption, (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (255, 0, 0), 1)


def main(args: argparse.Namespace, progress=None) -> None:
    if getattr(args, "scratch", False):
        detector = CNNDetector(device=args.device)
        captioner = VisionLanguageModel().to(args.device) if args.caption else None
    elif args.simple:
        detector = SimpleMotionDetector()
        captioner = SimpleCaptioner() if args.caption else None
    else:
        detector = YOLODetector(args.model, device=args.device)
        captioner = (
            CaptionGenerator() if args.caption and CaptionGenerator else None
        )

    scene_model = (
        SceneUnderstandingModel(detector, captioner)
        if getattr(args, "scene", False)
        else None
    )

    congestion = CongestionDetector()
    flamingo = (
        FlamingoVisionTextModel(detector, captioner)
        if getattr(args, "flamingo", False)
        else None
    )

    cap = cv2.VideoCapture(args.input)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = create_writer(args.output, fps, width, height)

    frame_index = 0
    log_lines = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if scene_model:
            result = scene_model.understand(frame)
            detections = result["detections"]
            caption = result["caption"]
        elif flamingo:
            detections, caption = flamingo.process(frame)
        else:
            detections = detector.detect(frame)
            if captioner:
                if hasattr(captioner, "caption"):
                    caption = captioner.caption(frame)
                else:
                    caption = captioner.generate(frame)
            else:
                caption = ""
        congested = congestion.update(detections)
        status = "Congested" if congested else "Free"

        annotate_frame(frame, detections, status, caption)
        writer.write(frame)
        log_lines.append(f"{frame_index},{status}\n")
        frame_index += 1
        if progress:
            progress()

    cap.release()
    writer.release()

    with open(args.log, "w") as f:
        f.writelines(log_lines)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Traffic congestion detection")
    parser.add_argument("--input", required=True, help="Input video path")
    parser.add_argument("--output", default="output.mp4", help="Output video path")
    parser.add_argument("--log", default="congestion_log.txt", help="Log file path")
    parser.add_argument("--model", default="yolov5s.pt", help="YOLO model weights")
    parser.add_argument("--device", default="cpu", help="Computation device")
    parser.add_argument("--caption", action="store_true", help="Enable caption generation")
    parser.add_argument(
        "--flamingo", action="store_true", help="Use Flamingo-inspired architecture"
    )
    parser.add_argument(
        "--simple",
        action="store_true",
        help="Use from-scratch detector and captioner",
    )
    parser.add_argument(
        "--scratch",
        action="store_true",
        help="Use deep models implemented from scratch",
    )
    parser.add_argument(
        "--scene",
        action="store_true",
        help="Use unified scene understanding model",
    )
    main(parser.parse_args())

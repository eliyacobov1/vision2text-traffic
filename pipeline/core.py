"""High-level vision-language pipeline components.

The :class:`VisionLanguagePipeline` coordinates detection, tracking and
caption generation in a single processing loop. Each stage exposes a small
interface so custom models can be inserted without rewriting the orchestration
logic. This design mirrors production systems where modularity eases research
experimentation while maintaining a clear data flow.
"""

from dataclasses import dataclass
from typing import List, Dict, Protocol


class Detector(Protocol):
    """Protocol for object detectors used in the pipeline."""

    def detect(self, frame) -> List[Dict]:
        ...


class Captioner(Protocol):
    """Protocol for caption generators."""

    def caption(self, frame) -> str:
        ...


@dataclass
class PipelineConfig:
    """Configuration for :class:`VisionLanguagePipeline`.

    Collecting parameters in a dataclass keeps experiments reproducible and
    allows the CLI to translate arguments directly into strongly typed fields.
    Models can be swapped by changing ``model`` or ``caption_model`` without
    altering the pipeline code.
    """

    input: str
    output: str = "output.mp4"
    log: str = "congestion_log.txt"
    model: str = "yolov5s.pt"
    caption_model: str = ""
    device: str = "cpu"
    caption: bool = True


class VisionLanguagePipeline:
    """Orchestrates detection, tracking and captioning.

    The pipeline loads the detector and a CLIP-based captioner lazily to avoid
    heavy dependencies during initialisation. Each frame flows through the
    Flamingo module which adds temporal context before generating a caption. This
    mirrors real systems that combine spatial detection with language models for
    scene understanding.
    """

    def __init__(self, config: PipelineConfig) -> None:
        from detector.yolo_detector import YOLODetector
        from analyzer.congestion_detector import CongestionDetector
        from flamingo.vision_text_model import FlamingoVisionTextModel
        try:
            from captioner.clip_captioner import CLIPCaptioner
        except Exception:  # pragma: no cover - optional dependency
            CLIPCaptioner = None  # type: ignore

        self.detector: Detector = YOLODetector(config.model, device=config.device)
        self.captioner: Captioner | None = None
        if config.caption and CLIPCaptioner:
            try:
                self.captioner = CLIPCaptioner(
                    model_path=config.caption_model, device=config.device
                )
            except Exception:  # pragma: no cover - failed to load model
                self.captioner = None
        self.congestion = CongestionDetector()
        self.flamingo = FlamingoVisionTextModel(self.detector, self.captioner)
        self.config = config

    def run(self, progress=None) -> None:
        import cv2
        from traffic_utils.video import create_writer

        cap = cv2.VideoCapture(self.config.input)
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        writer = create_writer(self.config.output, fps, width, height)

        frame_index = 0
        log_lines = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            detections, caption = self.flamingo.process(frame)

            congested = self.congestion.update(detections)
            status = "Congested" if congested else "Free"

            self._annotate_frame(frame, detections, status, caption)
            writer.write(frame)
            log_lines.append(f"{frame_index},{status}\n")
            frame_index += 1
            if progress:
                progress()

        cap.release()
        writer.release()
        with open(self.config.log, "w") as f:
            f.writelines(log_lines)

    @staticmethod
    def _annotate_frame(frame, detections: List[Dict], status: str, caption: str = "") -> None:
        import cv2

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

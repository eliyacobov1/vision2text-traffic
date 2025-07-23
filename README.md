# Vision2Text Traffic

A showcase of **vision‑language models** for traffic analysis. The system detects
and tracks vehicles, then uses a captioning model to describe the scene in
natural language. This single pipeline demonstrates how vision and language can
be combined for rich traffic understanding.

## Installation

Create a Python environment and install dependencies:

```bash
pip install -r requirements.txt
```

The detector uses YOLOv5 weights (`yolov5s.pt` by default). If not present
locally, the weights will be downloaded automatically using `torch.hub`.

## Usage

Run the main pipeline on a video file:

```bash
python main.py --input path/to/video.mp4 --output annotated.mp4 --log log.txt
```

The script performs YOLOv5 detection, aggregates recent frames with a
Flamingo-style module and generates a caption for each frame using a CLIP based
model. The result is an annotated video and a log of congestion status.

All configuration options are collected in a `PipelineConfig` dataclass so they
can easily be modified or extended in code.

To experiment with your own models, provide custom weights:

```bash
python main.py --input video.mp4 --model my_yolov5.pt \
    --caption-model /path/to/my_clip_weights
```

Use `--no-caption` to disable caption generation if needed (e.g. for testing).

The script outputs an annotated video, a log of congestion status for each frame, and optional captions overlayed on the frames.

## Quick Demo

Run a self-contained demonstration using a short sample video. The script
downloads the clip automatically and executes the end‑to‑end pipeline:

```bash
python demo.py --caption
```

After processing, the script reports how many frames were marked as congested and
the location of the generated video and log file.

## Architecture Overview

The processing logic lives in `VisionLanguagePipeline`, a class that wires
the detector, congestion analysis and caption generator together. Each module
follows a small interface, allowing you to swap in alternative detectors or
captioners with minimal code changes. Pipeline parameters are defined in
`PipelineConfig`, which can be extended for research experiments.


## Architecture Overview

Frames are processed by a YOLO detector to locate vehicles. Detected objects
feed into `FlamingoVisionTextModel`, which keeps a short history of frames and
aggregates them so the captioner sees temporal context. A CLIP-based captioner
then produces a brief description of the scene. Congestion is estimated from the
detection tracks and logged alongside the captions.

## Vision-Language Components

The pipeline is built from modular components:

- **YOLODetector** – fast object detector for locating vehicles.
- **CongestionDetector** – analyses tracked positions to estimate traffic flow.
- **CLIPCaptioner** – CLIP-style captioner producing short textual summaries.
- **FlamingoVisionTextModel** – maintains a context window of frames for the captioner.

These pieces can be replaced with your own models by supplying different weight files.

## Advanced Model Customisation

`main.py` accepts custom paths for both detection and captioning models. This allows you to drop in your own YOLO variants or CLIP weights without code changes. When selecting models consider the trade‑off between accuracy and runtime.

To experiment with your own models:

```bash
python main.py --input video.mp4 --model my_yolov5.pt \
    --caption-model ./my_clip_weights
```

The pipeline will automatically load the specified weights and integrate them into the detection‑tracking‑captioning loop.
For full control you can implement the minimal `Detector` or `Captioner`
protocols defined in the `pipeline` package and pass your objects directly to
`VisionLanguagePipeline`.

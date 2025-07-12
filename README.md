# Vision2Text Traffic

A modular system for detecting and analyzing traffic congestion in video footage and optionally generating natural-language descriptions.

## Installation

Create a Python environment and install dependencies:

```bash
pip install -r requirements.txt
```

The detector uses YOLOv5 weights (`yolov5s.pt` by default). If not present locally, the weights will be downloaded automatically using `torch.hub`.

Alternatively you can run a minimal implementation built entirely from scratch. Pass `--simple` to use a motion-based detector and a rule-based caption generator that require no deep learning dependencies. For a deep model also built from scratch, pass `--scratch` to use a tiny CNN detector and transformer captioner.

## Usage

Run the main pipeline on a video file:

```bash
python main.py --input path/to/video.mp4 --output annotated.mp4 --log log.txt
```

Run the pipeline without external deep learning models:

```bash
python main.py --input path/to/video.mp4 --output annotated.mp4 --log log.txt --simple
```

Add `--caption` to enable caption generation (requires additional model download).

The script outputs an annotated video, a log of congestion status for each frame, and optional captions overlayed on the frames.

## Scene Understanding Pipeline

This project demonstrates how detection, tracking and captioning interact to
build a basic scene understanding system for autonomous driving research. Video
frames are first processed by an object detector (YOLO or custom CNN) to locate
vehicles. The `CongestionDetector` then analyses object trajectories using a
Kalman filter and optical flow to estimate speed and density. Finally, a
vision-language model summarises the scene in natural language, enabling
high-level insights about traffic conditions.

## Flamingo-Inspired Mode

Enable a lightweight Flamingo-style module with `--flamingo`. The model maintains
a rolling window of recent frames and aggregates them using a weighted average so
newer frames influence the caption more than older ones. This temporal context
helps the captioner describe evolving scenes.

```bash
python main.py --input path/to/video.mp4 --output annotated.mp4 --log log.txt --flamingo --caption
```

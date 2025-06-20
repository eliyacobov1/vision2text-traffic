# Vision2Text Traffic

A modular system for detecting and analyzing traffic congestion in video footage and optionally generating natural-language descriptions.

## Installation

Create a Python environment and install dependencies:

```bash
pip install -r requirements.txt
```

The detector uses YOLOv5 weights (`yolov5s.pt` by default). If not present locally, the weights will be downloaded automatically using `torch.hub`.

## Usage

Run the main pipeline on a video file:

```bash
python main.py --input path/to/video.mp4 --output annotated.mp4 --log log.txt
```

Add `--caption` to enable caption generation (requires additional model download).

The script outputs an annotated video, a log of congestion status for each frame, and optional captions overlayed on the frames.

## Flamingo-Inspired Mode

Enable a lightweight Flamingo-style module with `--flamingo`. The model maintains
a rolling window of recent frames and aggregates them using a weighted average so
newer frames influence the caption more than older ones. This temporal context
helps the captioner describe evolving scenes.

```bash
python main.py --input path/to/video.mp4 --output annotated.mp4 --log log.txt --flamingo --caption
```

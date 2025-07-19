# Vision2Text Traffic

A showcase of **vision‑language models** for traffic analysis. The system detects and tracks vehicles, then uses a captioning model to describe the scene in natural language. This bridges traditional computer vision with text generation.

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
For a unified approach that analyses the full scene before captioning, pass `--scene`.

The script outputs an annotated video, a log of congestion status for each frame, and optional captions overlayed on the frames.

## Quick Demo

Run a self-contained demonstration using a short sample video. The script
downloads the clip automatically and executes the end‑to‑end pipeline. Three
modes are available:

* **full** – YOLOv5 detection with a transformer captioning model.
* **simple** – motion based detector with rule‑based captions.
* **scratch** – tiny CNN detector and transformer written from scratch.

```bash
# full pipeline with captions and Flamingo temporal context
python demo.py --mode full --caption --flamingo --scene

# lightweight demo that avoids heavy model downloads
python demo.py --mode simple
```

After processing, the script reports how many frames were marked as congested and
the location of the generated video and log file.

## Scene Understanding Pipeline

This project demonstrates how detection, tracking and captioning interact to
build a basic scene understanding system for autonomous driving research. Video
frames are first processed by an object detector (YOLO or custom CNN) to locate
vehicles. The `CongestionDetector` then analyses object trajectories using a
Kalman filter and optical flow to estimate speed and density. Finally, a
vision-language model summarises the scene in natural language, enabling
high-level insights about traffic conditions. For a more integrated approach,
the `SceneUnderstandingModel` can be enabled with `--scene` to combine detection
and captioning in one step.

## Flamingo-Inspired Mode

Enable a lightweight Flamingo-style module with `--flamingo`. The model maintains
a rolling window of recent frames and aggregates them using a weighted average so
newer frames influence the caption more than older ones. This temporal context
helps the captioner describe evolving scenes.

```bash
python main.py --input path/to/video.mp4 --output annotated.mp4 --log log.txt --flamingo --caption
```

## Vision-Language Components

Several captioning modules are included to explore different styles of vision‑language modelling:

- **CaptionGenerator** – wraps a pretrained ViT‑GPT2 model from Hugging Face for quick, high quality captions.
- **VisionLanguageModel** – a tiny transformer implementation written from scratch. Its `generate` method can optionally take detection bounding boxes to fuse object-centric features with global context.
- **FlamingoVisionTextModel** – aggregates a temporal window of frames before captioning to mimic the Flamingo architecture.
- **SceneUnderstandingModel** – combines detection and captioning in one step to summarise the whole scene.

These modules highlight the repository’s focus on combining visual understanding with textual descriptions.

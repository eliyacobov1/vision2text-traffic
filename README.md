# Vision-Language Traffic Congestion Detector

This project explores a compact vision-language model for scoring traffic congestion. A frozen Vision Transformer encodes the road image while DistilBERT embeds a short text prompt such as "heavy traffic" or "clear road." Their representations are fused with cross-attention to predict how well the image matches the prompt. When the contrastive head is activated, the model can also retrieve similar examples from the training set.

## Features
- Image–text classification with frozen encoders
- Optional contrastive retrieval head for nearest-neighbor search
- Configuration-driven training and evaluation
- Lightweight CLI demo for quick experimentation

## Architecture
### Encoders
- **Vision Transformer (ViT)** – turns the image into patch embeddings.
- **DistilBERT** – embeds the text prompt.

Both encoders stay frozen.

### Fusion
Two cross-attention blocks let text tokens attend to image tokens, producing fused representations that capture how the prompt relates to the scene.

### Heads
- **Classification head** – linear layer over the fused `[CLS]` token trained with binary cross-entropy to predict congestion.
- **Contrastive head** – projects averaged image tokens and the text `[CLS]` into a shared embedding space. An InfoNCE loss aligns matching image-text pairs and separates mismatched pairs, enabling retrieval.

## Data and Training
Training expects a CSV file with `image_url`, `text`, and `label` columns (see `sample_data/dataset.csv`). Images download on demand and cache locally. A YAML config file defines hyperparameters, augmentation, and whether the contrastive head is active. Checkpoints are written to `checkpoints/` and TensorBoard logs to `runs/`.

## Quick Start
### Setup
Install dependencies and run tests:

```bash
pip install -r requirements.txt
pytest
```

### Train a Model

```bash
python main.py train --config configs/config_classify.yaml
```

### Evaluate

```bash
python main.py eval --config configs/config_classify.yaml --ckpt checkpoints/model.pt
```

### Run the Demo

```bash
./run_demo.sh --ckpt checkpoints/model.pt --config config.yaml
```

### Example CLI Query
After training, obtain a congestion score and nearest example:

```bash
python cli.py --ckpt checkpoints/model.pt --image sample_data/jam.jpg --text "heavy traffic"
```

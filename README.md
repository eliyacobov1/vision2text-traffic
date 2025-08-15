# Vision-Language Traffic Congestion Detector

A small prototype that combines a frozen Vision Transformer and DistilBERT to score traffic congestion. The model accepts a road image and a short text prompt such as "heavy traffic" or "clear road" and returns a probability that the scene matches the prompt. When the contrastive head is enabled, it can also retrieve similar examples from the training data.

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
Training expects a CSV file with `image_url`, `text`, and `label` columns (see `sample_data/dataset.csv`). Images download on demand and cache locally. A YAML config file sets hyperparameters, augmentation, and whether the contrastive head is active. Checkpoints go to `checkpoints/` and TensorBoard logs to `runs/`.

## Usage
### Setup
Install dependencies and run tests:

```bash
pip install -r requirements.txt
pytest
```

### Train

```bash
python main.py train --config configs/config_classify.yaml
```

### Evaluate

```bash
python main.py eval --config configs/config_classify.yaml --ckpt checkpoints/model.pt
```

### Demo

```bash
./run_demo.sh --ckpt checkpoints/model.pt --config config.yaml
```

### Example CLI Query
After training, you can obtain a congestion score and nearest example:

```bash
python cli.py --ckpt checkpoints/model.pt --image sample_data/jam.jpg --text "heavy traffic"
```

## Limitations and Future Work
- Small dataset and limited tuning yield modest accuracy.
- Not optimized for production deployment.
- Future improvements could include fine-tuning the encoders, richer augmentations, and multi-level congestion labels.

## Project scope
This is a personal learning project that demonstrates how to wire together vision and language modules for basic traffic analysis.


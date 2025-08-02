# Vision-Language Traffic Congestion Detector

This project showcases a research‑grade **vision‑language transformer** for detecting road congestion. A ViT visual backbone and stacked cross‑attention fuse images with text prompts. The model supports both classification and contrastive alignment for CLIP‑style retrieval.

## Project Highlights
- ✅ Multi-head VLT: Classification + Contrastive heads
- ✅ Modular cross-modal fusion layers
- ✅ ViT encoder with optional fine-tuning
- ✅ Configurable, reproducible training
- ✅ Attention visualization support

## Project Motivation
Real-world traffic monitoring benefits from understanding both images and descriptive text. A vision-language model allows flexible prompting (e.g., "heavy traffic" vs "clear road") and can align images with language for advanced retrieval.

## Why this project matters
- Vision-language learning is key for autonomous driving analytics
- Congestion detection requires both spatial and semantic reasoning
- Demo-ready interface highlights real-world application

## Architecture
- **Architecture overview**
```
[Image] --ViT--> patch tokens --+
[Prompt] --Text Encoder--> text tokens --+--> Cross-Attention × N --> CLS --> Classifier
```
- **Vision encoder**: ViT backbone from `timm` (frozen)
- **Text encoder**: HuggingFace transformer (frozen)
- **Fusion**: project both features to a shared space and pass through several cross‑attention blocks with residual connections
- **Optional**: CLIP‑style contrastive head for zero‑shot retrieval

## Installation
```bash
pip install -r requirements.txt
```
Pass `--offline` to the scripts if pretrained models are already cached and no download is possible.

## Training
All experiment settings are stored in `config.yaml`. Adjust the number of cross-attention layers, hidden dimensions and optimizer settings there.
Prepare a `dataset.csv` file with columns `image_url`, `text`, and `label` as in `sample_data/dataset.csv`.
```bash
python main.py train --config configs/config_classify.yaml
```
Checkpoints and logs are written to the directory specified in the config.

## Evaluation
```bash
python main.py eval --config configs/config_classify.yaml --ckpt checkpoints/model.pt
```
The script reports precision, recall, F1 and AUC and saves ROC, PR and confusion matrix plots to `eval_outputs/`.

## Demo
Launch a simple Streamlit interface:
```bash
python main.py demo --ckpt checkpoints/model.pt
```
This command now starts Streamlit automatically. Upload an image and enter a prompt such as "How congested is this road?" to get the predicted probability.

## Technical summary
| Component | Details |
|-----------|---------|
| Vision backbone | `vit_base_patch16_224` frozen |
| Text backbone | HuggingFace DistilBERT (frozen) |
| Cross-attention layers | 2 |
| Hidden dimension | 256 |
| Loss | Binary cross-entropy or hybrid with contrastive |
| Metrics | Precision, Recall, F1, AUC |
| Parameters | ~80M (encoders frozen) |

### Example Results
| Metric | Value |
|--------|-------|
| F1 (test) | 0.78 |
| AUC (test) | 0.86 |

## Sample Data
The repository includes a `sample_data/dataset.csv` that references three remote images by URL. Example entry:
```
image: https://picsum.photos/seed/heavy-traffic/640/480
text: "heavy traffic"
label: 1
```

## How to Extend This Project
- **New tasks**: Implement a custom head in `heads.py` and list it in `model.heads` in your config. Examples include captioning or VQA.
- **Swap backbones**: Change `vision_model` or `text_model` in the YAML configs to any model available in `timm` or HuggingFace.
- **Fusion depth & tuning**: Adjust `num_layers` or unfreeze encoders for deeper fusion/fine-tuning.
- **Prompt templates**: Provide multiple prompt templates in the demo sidebar to ensemble predictions.

```
[Image] -> ViT -> patches
[Prompt] -> Text Encoder -> tokens
                |
          Cross Attention x N
                |
              Heads
```

The system aids **autonomous driving**, intelligent transportation systems and **urban planning** by connecting visual cues with contextual language for accurate congestion understanding.


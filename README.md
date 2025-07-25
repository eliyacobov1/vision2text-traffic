# Vision-Language Traffic Congestion Detector

This repository demonstrates a small **vision‑language transformer** for detecting road congestion. A vision‑language model processes an image together with a text prompt so that both modalities influence the prediction.

## Architecture
```
[Image]  --Vision Encoder-->  image features --+
[Prompt] --Text Encoder-->   text features  --+-->
                     Cross‑Attention
                           ↓
                     Classifier → congestion score
```
- **Vision encoder**: ResNet backbone from `torchvision` (frozen)
- **Text encoder**: HuggingFace transformer (frozen)
- **Fusion**: project both features to a shared space and apply a cross‑attention layer

## Installation
```bash
pip install -r requirements.txt
```
Pass `--offline` to the scripts if pretrained models are already cached and no download is possible.

## Training
Prepare a `dataset.csv` file with columns `image_url`, `text`, and `label`. The sample_data directory provides an example with online image links.
```bash
python train.py --data-dir sample_data --out-dir checkpoints --epochs 1
```
A checkpoint `model.pt` and `model_last.pt` will be written to the output directory.

## Evaluation
```bash
python evaluate.py --data-dir sample_data --ckpt checkpoints/model.pt
```
This prints precision, recall, F1 and AUC metrics together with a few example predictions.

## Demo
Launch a simple Streamlit interface:
```bash
./run_demo.sh
```
Upload an image and enter a prompt such as "How congested is this road?" to get the predicted probability.

## Sample Data
The repository includes a minimal `dataset.csv` with three examples referencing images hosted online. Example output:
```
Input URL: https://i.imgur.com/ExdKOOz.png
Prompt: "heavy traffic"
Prediction: 0.75 (75% congestion)
```

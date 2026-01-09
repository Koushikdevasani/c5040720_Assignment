Multimodal StoryReasoning — Visual + Text + Reasoning Tag Encoder for Diffusion-Based Image Decoding
This repository implements a multimodal temporal encoder for narrative understanding using the StoryReasoning dataset.
The model encodes:
Image sequences (frames)
Narrative captions
Reasoning tags (causal relations, object interactions, scene transitions)
and produces:
Contextual caption predictions, and
Frame-level visual embeddings suitable for conditioning a diffusion-based image decoder.
This project supports end-to-end data loading, preprocessing, training, evaluation, and visualization via both Python modules and a full Jupyter Notebook.
 Project Structure
project/
│
├── config.yaml                # configuration (dataset, model, training)
├── requirements.txt           # Python dependencies
├── README.md                  # you are here
│
├── src/
│   ├── model.py               # multimodal CNN/LSTM model with reasoning encoder
│   ├── train.py               # training script (standalone CLI)
│   ├── utils.py               # dataset loader, tokenizer, preprocessing
│
└── notebooks/
└── storyreasoning_multimodal.ipynb    # full EDA → training → evaluation pipeline
 Features
1.Multimodal inputs
The model fuses three streams per timestep:
Visual encoder — lightweight CNN → 512-dim features
Caption encoder — BERT-tokenized LSTM → 512-dim
Reasoning tag encoder — tokenized tag sequences → 256–512 dim
2.Temporal modeling
A story-level LSTM processes fused per-frame embeddings to learn:
narrative flow
temporal consistency
causal sequences
3.Outputs
The model produces:
next-caption token logits
learned image embedding → used to condition a future diffusion image decoder
4.Jupyter Notebook
The included notebook covers:
HF dataset loading
Visual EDA (frames + captions + reasoning tags)
Model building
Training with progress bars
Evaluation
Visualizing predictions
 Installation
Create a fresh environment:
python -m venv venv
source venv/bin/activate

Install dependencies:
pip install -r requirements.txt

For GPU:
pip install "tensorflow[and-cuda]>=2.14.0"
 Configuration (config.yaml)
Key options:
dataset:
hf_name: "daniel3303/StoryReasoning"
seq_len: 3
batch_size: 16
image_size: 128
max_caption_len: 32
max_reason_len: 32
model:
vocab_size: 30522
image_feat_dim: 512
text_embed_dim: 300
text_hidden_dim: 512
reason_embed_dim: 256
reason_hidden_dim: 512
multimodal_dim: 512
temporal_hidden_dim: 512
text_decoder_hidden: 512
pad_token_id: 0
bos_token_id: 101
eos_token_id: 102
training:
lr: 1e-4
epochs: 5
grad_clip: 1.0
log_interval: 50
save_dir: "results/checkpoints"
 Dataset
The project uses the HuggingFace dataset:
daniel3303/StoryReasoning

Each sample contains:
image frames
textual story segments
reasoning annotations
temporal ordering
The utils.py loader automatically detects fields such as:
frames, images, frames_paths, image_paths
captions, descriptions, texts, story
reasoning_tags, reasoning, logic_tags, reasons

and pads sequences to match seq_len.
 Model Training
You can train either via:
1.Command-line script
python src/train.py --config config.yaml

For quick debugging:
python src/train.py --config config.yaml --small
2.Jupyter Notebook (recommended)
Open:
notebooks/storyreasoning_multimodal.ipynb

The notebook walks through:
Data loading
EDA (visualizing frames + captions + reasoning tags)
Dataset preprocessing
Building the multimodal model
Full training loop with tqdm progress bar
Evaluation on test set
Visualizing predictions & caption reconstruction
 Evaluation & Metrics
The model supports the following outputs for downstream evaluation:
1.	Text reconstruction loss
Sparse cross-entropy on token-level predictions.
2.	Image embedding regression loss
MSE between predicted image feature and CNN-encoded ground truth frame.
3.	Optional downstream metrics
(Notebook-ready, not yet coded in main scripts)
CLIP text-image alignment
Structural similarity (SSIM)
Frame consistency score
FID for image quality (requires diffusion decoder outputs)
 Integration with a Diffusion Image Decoder
The multimodal model outputs:
img_pred ∈ R^(image_feat_dim)

This can condition a diffusion model via:
feature concatenation
adaptive layer normalization (AdaLN)
cross-attention
time-condition vector injection
We can provide a full training script for a DDPM / Stable Diffusion–style UNet if needed.
 Visualization
The notebook includes:
frame visualization
ground-truth vs predicted caption
training curves
loss plots
Example visualization:
Frame 0 | caption: "A boy opens a door..."
Frame 1 | caption: "He sees a small robot..."
Frame 2 | caption: "The robot gestures toward..."
 Contributing
Improvements welcome!
Add diffusion decoder
Implement CLIP-based story coherence evaluation
Optimize data pipeline
Add temporal attention instead of LSTM

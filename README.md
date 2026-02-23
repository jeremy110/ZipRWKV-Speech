# ZipRWKV-Speech (Research Stage)
[English] | [中文版](./README.zh-CN.md)

ZipRWKV-Speech is a Speech-LLM based Automatic Speech Recognition (ASR) system. Its core architecture utilizes **Zipformer** as the speech encoder, integrated with **RWKV7** as the language model backbone.

> [!IMPORTANT]

> **Research Phase Disclaimer**: This project is currently in the experimental/development stage and serves primarily as a personal implementation record. The codebase draws from several open-source projects (such as NeMo, K2, and RWKV), extracting and simplifying key components to create a lightweight yet high-performance SpeechLLM framework. Each module contains its own README to explain the implementation logic.

---

## 🏗️ System Architecture
* **Speech Encoder:** Zipformer (from K2/Icefall), providing efficient downsampling and feature extraction.

* **LLM Backbone:** RWKV7, combining the inference efficiency of RNNs with the training performance of Transformers.

* **Data Pipeline:** A dynamic bucketing system based on Lhotse.

## 🚀 Development Roadmap
Current progress on code organization and implementation:

- [x] **Data Pipeline (Lhotse-based)**
    - [x] Support for NeMo-style Manifest loading.
    - [x] Implementation of `DynamicBucketingSampler` for dynamic Batch Size adjustment.
    - [x] Integration of `Cutset.mux` for weighted multi-source data mixing.
    - [x] On-the-fly data augmentation (Speed, Volume, Noise, SpecAugment).
    - [x] test dataset code and `conf.yaml`.
    - [x] noise manifest prepare script.
- [ ] **Model Architecture**
    - [X] Zipformer Encoder integration and test code.
    - [ ] RWKV7 and PEFT (Parameter-Efficient Fine-Tuning) integration.
- [ ] **Training Implementation**
    - [ ] PyTorch Lightning Training Module.
- [ ] **Checkpoints & Evaluation**
    - [ ] Release of pre-trained model weights.

<details>
<summary><h3> K2 Installation Guide </h3></summary>

## Environment Requirements

- Python 3.10
- CUDA 12.8
- PyTorch 2.8

## Installation Steps

### 1. Download K2 Wheel

Download the [K2 wheel](https://k2-fsa.github.io/k2/cuda.html) file that matches your environment configuration:

```bash
wget https://k2-fsa.github.io/k2/cuda.html
wget k2-1.24.4.dev20250807+cuda12.8.torch2.8.0-cp310-cp310-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl
```

### 2. Fix the Downloaded Filename

Sometimes the `+` character in the filename may be converted to `%2B` after download. You need to rename it back to `+`:

```bash
# If the filename contains %2B, change it to +
# Original filename: k2-1.24.4.dev20250807%2Bcuda12.8.torch2.8.0-cp310-cp310-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl
# New filename: k2-1.24.4.dev20250807+cuda12.8.torch2.8.0-cp310-cp310-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl
```

### 3. Install K2

Install the wheel package using `uv pip`:

```bash
uv pip install "k2-1.24.4.dev20250807+cuda12.8.torch2.8.0-cp310-cp310-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl"
```
</details>

## Zipformer Encoder

This project uses [`AudenAI/auden-encoder-tta-m10`](https://huggingface.co/AudenAI/auden-encoder-tta-m10) as the encoder for Zipformer.

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
    - [ ] test code and `conf.yaml`.
- [ ] **Model Architecture**
    - [ ] Zipformer Encoder integration.
    - [ ] RWKV7 and PEFT (Parameter-Efficient Fine-Tuning) integration.
- [ ] **Training Implementation**
    - [ ] PyTorch Lightning Training Module.
- [ ] **Checkpoints & Evaluation**
    - [ ] Release of pre-trained model weights.
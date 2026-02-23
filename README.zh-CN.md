# ZipRWKV-Speech (Research Stage)

[English](./README.md) | [中文版]

這個專案是一個基於 LLM 的語音辨識 (SpeechLLM)，核心架構採用 **Zipformer** 作為 Speech Encoder，並結合 **RWKV7** 作為 Language Model。

> [!IMPORTANT]  
> **研究階段說明**：本專案目前處於開發與實驗階段，主要目的是紀錄個人實作過程。程式碼參考了多個開源專案（如 NeMo, K2, RWKV），擷取片段並進行簡化與重組，旨在打造一個輕量且高效的 SpeechLLM 框架，每個部分會各有一個 README 來大致說明寫法。

---

## 🏗️ 系統架構 (System Architecture)

* **Speech Encoder:** Zipformer (來自 K2/Icefall)，提供高效的下採樣與特徵提取。
* **LLM Backbone:** RWKV7 ，結合 RNN 的推理效率與 Transformer 的訓練表現。
* **Data Pipeline:** 基於 Lhotse 的動態分桶 (Dynamic Bucketing) 系統。

---

## 🚀 開發進度與路線圖 (Roadmap)

目前的程式整理進度如下，持續更新中：

- [x] **Data Pipeline (Lhotse-based)**
    - [x] 支持 NeMo 格式 Manifest 讀取。
    - [x] 實現 `DynamicBucketingSampler` 動態 Batch Size 調整。
    - [x] 整合 `Cutset.mux` 權重化多數據源混合。
    - [x] 在線數據增強 (Speed, Volume, Noise, SpecAugment)。
    - [x] test dataset code and `conf.yaml`.
    - [x] noise manifest prepare script.
- [ ] **Model Architecture**
    - [X] Zipformer Encoder 整合 and test code。
    - [ ] RWKV7 以及 peft 整合。
- [ ] **Training Implementation**
    - [ ] PyTorch Lightning Training Module。
- [ ] **Checkpoints & Evaluation**
    - [ ] 提供預訓練模型權重。

---


<details>
<summary><h3> K2 安裝指南 </h3></summary>

## 環境要求

- Python 3.10
- CUDA 12.8
- PyTorch 2.8

## 安裝步驟

### 1. 下載 K2 whl

根據你的環境配置，從官方 [K2 CUDA 頁面](https://k2-fsa.github.io/k2/cuda.html) 下載正確版本的 whl：

```bash
wget https://huggingface.co/csukuangfj/k2/resolve/main/ubuntu-cuda/k2-1.24.4.dev20250807+cuda12.8.torch2.8.0-cp310-cp310-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl
```

### 2. 修復下載後的文件名

有時候下載完的文件名中的 `+` 號會被轉換為 `%2B`，你需要手動改回 `+`：

```bash
# 如果文件名包含 %2B，將其改為 +
# 原文件名: k2-1.24.4.dev20250807%2Bcuda12.8.torch2.8.0-cp310-cp310-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl
# 新文件名: k2-1.24.4.dev20250807+cuda12.8.torch2.8.0-cp310-cp310-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl
```

### 3. 安裝 K2

使用 `uv pip` 安裝下載好的 whl：

```bash
uv pip install "k2-1.24.4.dev20250807+cuda12.8.torch2.8.0-cp310-cp310-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl"
```
</details>

## Zipformer speech encoder

本項目使用 [`AudenAI/auden-encoder-tta-m10`](https://huggingface.co/AudenAI/auden-encoder-tta-m10)。


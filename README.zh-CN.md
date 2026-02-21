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
    - [ ] test code and `conf.yaml`.
- [ ] **Model Architecture**
    - [ ] Zipformer Encoder 整合。
    - [ ] RWKV7 以及 peft 整合。
- [ ] **Training Implementation**
    - [ ] PyTorch Lightning Training Module。
- [ ] **Checkpoints & Evaluation**
    - [ ] 提供預訓練模型權重。

---
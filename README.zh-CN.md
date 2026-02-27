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
- [X] **Model Architecture**
    - [X] Zipformer Encoder 整合 and test code。
    - [X] RWKV7 以及 peft 整合。
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

<details>
<summary><h3> LLM-Based ASR：音頻與文本嵌入融合概念 </h3></summary>

## 核心概念

![merge_features](./images/merge_features.jpg)

### 概念 1：音頻下採樣與 token 對齊

**編碼器輸出到 token**

在音頻編碼（Fbank → 編碼器）後，編碼器輸出特徵通過**投影層**進行下採樣：

- **編碼器輸出形狀**：`(B, T, encoder_out_dim)`，其中 T = 音頻秒數 × 25
- **投影層下採樣後**：`(B, T//2, lim_dim)`
- **時間解析度**：約 12.5 Hz
  - 每個 token 代表 **80ms** 的音頻
  - **一個中文字持續時間**：0.2~0.3 秒
  - **每個字的 token**：2~4 個 token


### 概念 2：提示詞設計與 LLM 兼容性

**特定於模型的提示詞格式**

不同的預訓練 LLM 模型需要不同的提示詞結構，這些結構在訓練時已經定義：

- **RWKV7**：簡單格式，如 `User: ` 和 `Assistant:`
- **Qwen**：特殊 token，如 `<|im_start|>` 和 `<|im_end|>`
- **ASR 特定提示詞**：常見的中文提示詞包括「請轉錄這段{lang}語音」

補充信息：一些論文也提出了提示詞投影模塊來解決提示詞的影響。[Reducing Prompt Sensitivity in LLM-based Speech Recognition Through Learnable Projection](https://arxiv.org/html/2601.20898v1)

**提示詞投影模塊替代方案**

[近期論文](https://arxiv.org/html/2409.19510v2)提出使用更簡單的方法在實踐中也效果很好：

- 簡單的特殊 token，如 `<|en|>`（或中文的 `<|zh|>`）通常就足夠了
- 這種輕量級方法降低了計算開銷，同時保持了有效性
- 語言規範幫助模型適當調整其解碼策略

### 概念 3：融合與 LLM 前向傳遞

**嵌入拼接策略**

整個系統的關鍵是簡單的拼接：

1. **下採樣的音頻特徵**：形狀 `(B, T//2, lim_dim)`
2. **文本 embeddings**：Various prompt tokens and assistant markers
3. **拼接**：將音頻特徵放在 `User: ` embeddings 的後面
4. **合併輸入**：`[User_emb] + [Audio_tokens] + [Assistant_emb] + [Label_emb]`，總形狀為 `(B, seq_len, lim_dim)`

**訓練 vs. 推理的區別**

- **訓練**：
  - 拼接橘色部分 + 綠色部分
  - 將整個序列送入 LLM 進行前向傳遞
  - 計算綠色部分的損失以訓練模型

- **推理**：
  - 只需要橘色部分
  - 綠色部分由 LLM 的自迴歸解碼生成
  - 無需在推理時提供目標文本



## 總結

這種方法的優雅之處在於其簡潔性：

1. **下採樣**生成自然對齊的 token（每個約 80ms），與音素單位相對應
2. **提示詞設計**簡潔明了且依賴於模型；語言標識符就足夠了
3. **融合**只是按特定順序拼接 embeddings
4. **訓練/推理不對稱性**利用了 LLM 固有的自迴歸文本生成能力

通過將下採樣的音頻特徵與精心格式化的文本提示相結合，該系統使 LLM 能夠在沒有複雜架構修改的情況下執行有效的端到端 ASR。

</details>

## 🙏 Acknowledgements
We borrowed a lot of code from the following excellent projects:
- [rkwv7](https://github.com/BlinkDL/RWKV-LM)
- [rwkv_asr](https://huggingface.co/yueyulin/rwkv_asr)
- [speechllm](https://github.com/zhu-han/SpeechLLM)
- [Auden](https://github.com/AudenAI/Auden)
- [Zipformer_Lightning](https://github.com/ZQuang2202/Zipformer_Lightning)
- [icefall](https://github.com/k2-fsa/icefall)
- [RWKV-PEFT](https://github.com/Joluck/RWKV-PEFT)
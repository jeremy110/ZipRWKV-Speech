# ASR / AST DataLoader 模組說明

[English](./README.md) | [中文版]

本專案實作一套基於 [Lhotse](https://github.com/lhotse-speech/lhotse) 與 [PyTorch Lightning](https://lightning.ai/) 的語音辨識（ASR）與語音翻譯（AST）資料載入流程，透過動態 batch size 策略提升 GPU 利用效率 (參考 nemo 以及 k2 的寫法)。

---

## 目錄結構

```
.
├── nemo_adapters.py       # 讀取 manifest.json，轉換為 Lhotse Cut 格式
├── datamodule.py          # CustomDataset：資料集初始化與 __getitem__ 處理
└── dataloader_module.py   # Lightning DataModule：train/val dataloader 定義
```

---

## Manifest 格式

每行一筆 JSON，支援 ASR 與 AST 任務，預留 `ast_text` 與語言 tag 欄位：

```json
{"audio_filepath": "/path/to/a.wav", "duration": 5.191, "offset": 0, "asr_text": "", "ast_text": "", "source_lang": "zh", "target_lang": "zh"}
{"audio_filepath": "/path/to/b.wav", "duration": 4.417, "offset": 0, "asr_text": "", "ast_text": "", "source_lang": "zh", "target_lang": "zh"}
```

| 欄位 | 說明 |
|------|------|
| `audio_filepath` | 音檔絕對路徑 |
| `duration` | 音檔長度（秒） |
| `offset` | 讀取起始位置（秒） |
| `asr_text` | ASR 轉錄文字 |
| `ast_text` | AST 翻譯文字 |
| `source_lang` | 來源語言 tag |
| `target_lang` | 目標語言 tag |

> 建議在前處理階段將所有音檔統一轉為 16kHz，`datamodule.py` 的 sample rate 轉換步驟可視情況省略。

---

## 模組說明

### `nemo_adapters.py`

參考 NeMo 的 manifest 讀取邏輯並加以簡化，將 manifest.json 中的每筆資料轉換為 Lhotse 的 `Cut` 物件，供後續 Sampler 使用。

**主要功能：**
- 解析 `audio_filepath`、`duration`、`offset`
- 將 `asr_text`、`ast_text`、`source_lang`、`target_lang` 存入 Cut 的 custom fields

---

### `datamodule.py`

定義 `CustomDataset`，繼承 `torch.utils.data.Dataset`，負責資料的讀取與預處理。

#### `__init__` 階段

1. **多資料集混合**：若有多個不同語言的 manifest，使用 `CutSet.mux()` 依照權重混合多個資料集。
2. **長度過濾**：依照 `max_length` 與 `min_length`（單位：秒）過濾不符合條件的音檔。

#### `__getitem__` 階段

1. **Sample Rate 轉換**（可選）：若音檔未統一為 16kHz，在此轉換。建議預處理時統一，此步驟可省略。
2. **資料增強**：
   - 速度增強（Speed Perturbation）(from k2)
   - 音量增強（Volume Perturbation）(from k2)
   - 加入背景噪音（Noise Augmentation）
3. **特徵擷取**：依照 Speech Encoder 所需擷取對應特徵；Zipformer 使用 **Fbank**，可參考 [Auden lhotse_datamodule.py](https://github.com/AudenAI/Auden/blob/main/src/auden/data/lhotse_datamodule.py#L112-L132) 調整。
4. **頻譜增強**（SpecAugment）：ASR 模型標準前處理，遮蔽時間軸與頻率軸。
5. **Feature Lengths 計算**：計算 `feature_lens`，提供給 Attention 機制處理 padding。
6. **文字與語言 Tag 處理**：依任務類型（ASR / AST）取出對應的 `asr_text` / `ast_text`，並附加 `source_lang` / `target_lang` tag。

---

### `dataloader_module.py`

基於 PyTorch Lightning 框架，提供標準命名的 `train_dataloader` 與 `val_dataloader`。

#### 動態 Batch Size（DynamicBucketingSampler）

使用 Lhotse 的 `DynamicBucketingSampler`，根據音檔時長動態組合 batch，避免短音檔浪費 padding、長音檔 OOM 的問題。

**關鍵參數：**

| 參數 | 說明 |
|------|------|
| `max_duration` / `train_max_duration` | 單一 batch 的最大總時長（秒），依 GPU 顯存調整 |
| `max_cuts` | 如果音檔長度普遍偏短，此值建議設定 32 以防 GPU OOM |

**其餘參數:**
可以照預設即可
| 參數 | 說明 |
|------|------|
| `num_buckets` | 分桶數量，影響音檔長度分佈的均勻度 |
| `buffer_size` | 越大，則 sample 的多樣性更好 |


**`train_max_duration` 設定建議：**
後續會測試 0.4B 的 LLM 在 RTX 4090 24G 以及 PRO 6000 96G 的數據
| GPU 顯存 | 建議值 |
|----------|--------------|
| 24 GB |  秒 |
| 96 GB |  秒 |

> 實際數值依模型大小與特徵維度而定，建議從較小值開始測試，逐步調高至不 OOM 為止。

---

## 資料流總覽

```
manifest.json
     │
     ▼
nemo_adapters.py
（轉為 Lhotse CutSet）
     │
     ▼
datamodule.py - CustomDataset.__init__
（多資料集混合 + 長度過濾）
     │
     ▼
dataloader_module.py
（DynamicBucketingSampler → torch DataLoader）
     │
     ▼
CustomDataset.__getitem__
（增強 → 特徵擷取 → SpecAugment → 文字處理）
     │
     ▼
Lightning Training Loop
```

---

## 參考資源

- [Auden lhotse_datamodule.py](https://github.com/AudenAI/Auden/blob/main/src/auden/data/lhotse_datamodule.py#L112-L132)
- [NeMo lhotse](https://github.com/NVIDIA/NeMo/blob/main/nemo/collections/common/data/lhotse/nemo_adapters.py#L46)
- [PyTorch Lightning DataModule](https://lightning.ai/docs/pytorch/stable/data/datamodule.html#what-is-a-datamodule)
- [zipformer lightning](https://github.com/ZQuang2202/Zipformer_Lightning/blob/main/zipformer_lightning/dataset/dataloader_module.py)
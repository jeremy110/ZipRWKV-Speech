# Audio to Lhotse Manifest Converter

將音訊目錄轉換為 [Lhotse](https://github.com/lhotse-speech/lhotse) CutSet Manifest 的工具腳本。  
A utility script that converts an audio directory into a [Lhotse](https://github.com/lhotse-speech/lhotse) CutSet Manifest.

---

## 使用方式 / Usage

### 使用 `uv run` 執行 / Run with `uv run`

```bash
uv run noise_manifest_prep.py \
  --audio_dir /path/to/wav/files \
  --output_dir /path/to/output \
  --prefix noise
```

---

## 參數說明 / Arguments

| 參數 / Argument | 必填 / Required | 預設值 / Default | 說明 / Description |
|---|---|---|---|
| `--audio_dir` | ✅ | — | 包含來源 `.wav` 檔案的根目錄 / Root directory containing source `.wav` files |
| `--output_dir` | ✅ | — | 輸出 manifest 的儲存目錄 / Directory where the generated manifest will be saved |
| `--prefix` | ❌ | `noise` | 輸出檔名的前綴 / Prefix for the output filename |

### 輸出檔案 / Output File

```
{output_dir}/{prefix}_cuts.jsonl.gz
```

---

## 邏輯說明 / Logic Walkthrough

腳本的核心流程分為三個步驟，將散落的音訊檔案統整為一個壓縮的 manifest。  
The core pipeline consists of three steps, consolidating scattered audio files into a single compressed manifest.

```
.wav files  ──►  RecordingSet  ──►  CutSet  ──►  manifest.jsonl.gz
```

### Step 1：建立 RecordingSet / Build RecordingSet

```python
recordings = RecordingSet.from_recordings(
    Recording.from_file(file) for file in audio_path.rglob("*.wav")
)
```

遞迴掃描 `--audio_dir` 底下所有 `.wav` 檔案，將每個檔案封裝成 `Recording` 物件（記錄路徑、取樣率、時長等 metadata），再組成一個 `RecordingSet`。  
Recursively scans all `.wav` files under `--audio_dir`. Each file is wrapped into a `Recording` object (storing path, sample rate, duration, etc.), then collected into a `RecordingSet`.

---

### Step 2：轉換為 CutSet / Convert to CutSet

```python
cuts = CutSet.from_manifests(recordings=recordings)
```

`Cut` 是 Lhotse 中最基本的操作單元，代表一段有明確時間範圍的音訊片段。此步驟將每筆 `Recording` 直接對應為一個涵蓋完整音訊的 `MonoCut`，方便後續進行切片、篩選、特徵擷取等操作。  
A `Cut` is the fundamental unit in Lhotse, representing an audio segment with explicit time boundaries. This step maps each `Recording` to a `MonoCut` spanning the full audio, enabling downstream operations like slicing, filtering, and feature extraction.

---

### Step 3：匯出為壓縮 JSONL / Export to Compressed JSONL

```python
cuts.to_file(output_path / f"{prefix}_cuts.jsonl.gz")
```

將 `CutSet` 的所有 metadata 序列化並以 gzip 壓縮格式寫出，產生輕量且易於分享的 manifest 檔案，可直接被 Lhotse 或下游訓練框架（如 k2 / NeMo）讀取。  
Serializes all `CutSet` metadata into a gzip-compressed JSONL file — a lightweight, portable manifest that can be directly loaded by Lhotse or downstream training frameworks (e.g., k2 / NeMo).

---

## 完整範例 / Full Example

```bash
uv run noise_manifest_prep.py \
  --audio_dir ./data/raw_audio \
  --output_dir ./manifests \
  --prefix train_noise
```

執行後會在 `./manifests/` 產生 `train_noise_cuts.jsonl.gz`。  
After execution, `train_noise_cuts.jsonl.gz` will be generated under `./manifests/`.

```
✅ Success! Manifest saved to: manifests/train_noise_cuts.jsonl.gz
📊 Total cuts processed: 1024
```

---

## 依賴套件 / Dependencies

```bash
uv add lhotse
```
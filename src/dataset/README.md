# ASR / AST DataLoader Module

[English] | [中文版](./README.zh-CN.md)

A [Lhotse](https://github.com/lhotse-speech/lhotse) + [PyTorch Lightning](https://lightning.ai/) data pipeline for Automatic Speech Recognition (ASR) and Automatic Speech Translation (AST), featuring dynamic batch size via `DynamicBucketingSampler` for efficient GPU utilization(Refer to the code of nemo and k2.).

---

## File Structure

```
.
├── nemo_adapters.py       # Parse manifest.json and convert to Lhotse Cut format
├── datamodule.py          # CustomDataset: initialization and __getitem__ preprocessing
└── dataloader_module.py   # Lightning DataModule: train/val dataloader definitions
```

---

## Manifest Format

One JSON object per line. Supports both ASR and AST tasks, with reserved fields for `ast_text` and language tags:

```json
{"audio_filepath": "/path/to/a.wav", "duration": 5.191, "offset": 0, "asr_text": "", "ast_text": "", "source_lang": "zh", "target_lang": "zh"}
{"audio_filepath": "/path/to/b.wav", "duration": 4.417, "offset": 0, "asr_text": "", "ast_text": "", "source_lang": "zh", "target_lang": "zh"}
```

| Field | Description |
|-------|-------------|
| `audio_filepath` | Absolute path to the audio file |
| `duration` | Audio duration in seconds |
| `offset` | Start offset for reading in seconds |
| `asr_text` | ASR transcription text |
| `ast_text` | AST translation text |
| `source_lang` | Source language tag |
| `target_lang` | Target language tag |

> It is recommended to resample all audio to 16kHz during preprocessing, in which case the sample rate conversion step in `datamodule.py` can be removed.

---

## Module Details

### `nemo_adapters.py`

Simplified from NeMo's manifest reading logic. Converts each entry in `manifest.json` into a Lhotse `Cut` object for use by the downstream sampler.

**Key responsibilities:**
- Parse `audio_filepath`, `duration`, `offset`
- Store `asr_text`, `ast_text`, `source_lang`, `target_lang` in the Cut's custom fields

---

### `datamodule.py`

Defines `CustomDataset`, a `torch.utils.data.Dataset` subclass responsible for data loading and preprocessing.

#### `__init__` stage

1. **Multi-dataset mixing**: When multiple manifests (e.g., different languages) are provided, `CutSet.mux()` is used to mix them according to specified weights.
2. **Duration filtering**: Filters out audio files that do not fall within `min_length` and `max_length` (in seconds).

#### `__getitem__` stage

1. **Sample rate conversion** (optional): Converts audio to the target sample rate if not done during preprocessing. Can be skipped if all audio is pre-resampled to 16kHz.
2. **Data augmentation**:
   - Speed perturbation (from k2)
   - Volume perturbation (from k2)
   - Noise augmentation (additive background noise)
3. **Feature extraction**: Extracts features based on the speech encoder's requirements. For Zipformer, **Fbank** features are used. See [Auden lhotse_datamodule.py](https://github.com/AudenAI/Auden/blob/main/src/auden/data/lhotse_datamodule.py#L112-L132) for reference.
4. **SpecAugment**: Standard spectrogram masking along both time and frequency axes.
5. **Feature length calculation**: Computes `feature_lens` to handle padding in attention mechanisms.
6. **Text and language tag processing**: Retrieves `asr_text` or `ast_text` according to task type, along with `source_lang` / `target_lang` tags.

---

### `dataloader_module.py`

A PyTorch Lightning-compatible module providing the standard `train_dataloader` and `val_dataloader` methods.

#### Dynamic Batch Size (DynamicBucketingSampler)

Uses Lhotse's `DynamicBucketingSampler` to group audio cuts into batches by total duration, avoiding wasted padding from short utterances and OOM errors from long ones.

**Key parameters:**

| Parameter | Description |
|-----------|-------------|
| `max_duration` / `train_max_duration` | Maximum total duration (seconds) per batch; tune based on GPU memory |
| `max_cuts` | If audio files are generally short, it is recommended to set this value to 32 to prevent GPU OOM errors. |

**Other parameters:** 
These can be set as default.
| Parameter | Description |
|-----------|-------------|
| `num_buckets` | Number of duration buckets; affects how evenly audio lengths are distributed |
| `buffer_size` | A larger `buffer_size` results in better sample diversity.|


**`train_max_duration` guidelines:**

I will subsequently test the settings of 0.4B LLM on the RTX 4090 24G and PRO 6000 96G.
| GPU VRAM | Suggested value |
|----------|-------------------------------|
| 24 GB | s |
| 40 GB | s |
| 80 GB | s |

> Actual values depend on model size and feature dimensions. Start conservatively and increase until just before OOM.

---

## Data Flow Overview

```
manifest.json
     │
     ▼
nemo_adapters.py
(Convert to Lhotse CutSet)
     │
     ▼
datamodule.py - CustomDataset.__init__
(Multi-dataset mixing + duration filtering)
     │
     ▼
dataloader_module.py
(DynamicBucketingSampler → torch DataLoader)
     │
     ▼
CustomDataset.__getitem__
(Augmentation → Feature extraction → SpecAugment → Text processing)
     │
     ▼
Lightning Training Loop
```

---

## References

- [Auden lhotse_datamodule.py](https://github.com/AudenAI/Auden/blob/main/src/auden/data/lhotse_datamodule.py#L112-L132)
- [NeMo lhotse](https://github.com/NVIDIA/NeMo/blob/main/nemo/collections/common/data/lhotse/nemo_adapters.py#L46)
- [PyTorch Lightning DataModule](https://lightning.ai/docs/pytorch/stable/data/datamodule.html#what-is-a-datamodule)
- [zipformer lightning](https://github.com/ZQuang2202/Zipformer_Lightning/blob/main/zipformer_lightning/dataset/dataloader_module.py)
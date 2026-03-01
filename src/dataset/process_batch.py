import torch
from torch.nn.utils.rnn import pad_sequence


def prepare_text_batch(
    asr_texts: list[str],
    ast_texts: list[str],
    languages: list[str],
    tokenizer,
    device: torch.device,
    eos_token_id: int = 0,
    pad_token_id: int = 0,
    mode: str = 'asr',
) -> dict[str, torch.Tensor]:
    """
    Tokenize and pad a batch of ASR/AST samples for LLM input.

    Training format  : User: [audio]<|en|>\nAssistant:Hello[eos]
    Inference format : User: [audio]<|en|>\nAssistant:

    Args:
        asr_texts:     Source transcription texts.
        ast_texts:     Translation texts (used in ast mode).
        languages:     Language tags in 'src_tgt' format, e.g. 'en_zh'.
        tokenizer:     rwkv tokenizer.
        device:        Target torch device.
        eos_token_id:  Token ID appended to label sequences.
        pad_token_id:  Token ID used for padding.
        mode:          'asr' or 'ast'.

    Returns:
        Dictionary with keys:
            text_input_ids      – prefix token IDs         (B, T_prefix)
            hints_ids           – lang prompt token IDs    (B, T_hints)
            text_attention_mask – attention mask for prefix (B, T_prefix)
            labels              – label token IDs          (B, T_label)
            labels_attention_mask – mask for labels        (B, T_label)
    """
    if mode not in ('asr', 'ast'):
        raise ValueError(f"Unsupported mode '{mode}'. Expected 'asr' or 'ast'.")

    # ── tokens that are identical for every sample in the batch ──────────────
    role_prefix_tokens = tokenizer.encode('User: ')
    role_prefix_len    = len(role_prefix_tokens)

    prefix_tokens_list:    list[list[int]] = []
    lang_prompt_tokens_list: list[list[int]] = []
    label_tokens_list:     list[list[int]] = []

    for asr_text, ast_text, language in zip(asr_texts, ast_texts, languages):
        src, tgt = language.split('_')

        if mode == 'asr':
            lang_prompt_tokens = tokenizer.encode(f'<|{src}|>\nAssistant:')
            label_tokens       = tokenizer.encode(asr_text) + [eos_token_id]
        else:  # mode == 'ast'
            lang_prompt_tokens = tokenizer.encode(f'<|{src}|><|{tgt}|>\nAssistant:')
            label_tokens       = tokenizer.encode(ast_text) + [eos_token_id]

        prefix_tokens_list.append(role_prefix_tokens)
        lang_prompt_tokens_list.append(lang_prompt_tokens)
        label_tokens_list.append(label_tokens)

    # ── convert to tensors ────────────────────────────────────────────────────
    def to_tensor(token_list: list[int]) -> torch.Tensor:
        return torch.tensor(token_list, dtype=torch.long, device=device)

    prefix_tensors      = [to_tensor(t) for t in prefix_tokens_list]
    lang_prompt_tensors = [to_tensor(t) for t in lang_prompt_tokens_list]
    label_tensors       = [to_tensor(t) for t in label_tokens_list]

    # attention mask has the same shape as the prefix — build directly from length
    prefix_attn_tensors = [
        torch.ones(role_prefix_len, dtype=torch.long, device=device)
        for _ in prefix_tensors
    ]

    # ── left-pad all sequences ────────────────────────────────────────────────
    pad_kwargs = dict(batch_first=True, padding_side='left')

    text_input_ids       = pad_sequence(prefix_tensors,      padding_value=pad_token_id, **pad_kwargs)
    hints_ids            = pad_sequence(lang_prompt_tensors, padding_value=pad_token_id, **pad_kwargs)
    labels               = pad_sequence(label_tensors,       padding_value=-100,         **pad_kwargs)
    text_attention_mask  = pad_sequence(prefix_attn_tensors, padding_value=0,            **pad_kwargs)
    labels_attention_mask = (labels != -100).long()

    return {
        'text_input_ids':       text_input_ids,
        'hints_ids':            hints_ids,
        'text_attention_mask':  text_attention_mask,
        'labels':               labels,
        'labels_attention_mask': labels_attention_mask,
    }


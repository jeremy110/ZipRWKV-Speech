import os
from dataclasses import dataclass

import torch

from peft import PeftModel, get_peft_model, MissConfig, TaskType

from models.rwkv.config import RWKVConfig
from models.rwkv.rwkv7 import RWKV7ModelForCausalLLMCuda


@dataclass
class RWKVArgs:
    n_layer: int
    n_embd: int
    head_size_a: int
    head_size_divisor: int
    dropout: float = 0.0
    vocab_size: int = None
    need_init_tmix: bool = False
    need_init_cmix: bool = False
    grad_cp: float = 0.0
    peft_r: int = 8

def load_encoder(
    cfg,
    dtype=torch.bfloat16,
    device="cuda",
):
    encoder = cfg["speech_encoder"]
    encoder_path = cfg["base_model"]["encoder_path"]
    ext = os.path.splitext(encoder_path)[1].lower()
    if ext == ".safetensors":
        from safetensors.torch import load_file as safe_load_file
        state_obj = safe_load_file(encoder_path, device = "cpu")
        state_dict = (
            state_obj["state_dict"]
            if isinstance(state_obj, dict) and "state_dict" in state_obj
            else state_obj
        )
        encoder.load_state_dict(state_dict, strict=True)
        encoder = encoder.to(dtype = dtype, device = device)
    else:
        state_dict = torch.load(cfg["base_model"]["encoder_path"], map_location = "cpu")
        encoder.load_state_dict(state_dict["model"], strict = False)
    
    encoder = encoder.to(dtype = dtype, device = device)

    return encoder

def load_projector(
    cfg,
    dtype = torch.bfloat16,
    device = "cuda",
):
    encoder_projector = cfg["encoder_projector"]
    if cfg["base_model"]["projector_path"] != None:
        state_dict = torch.load(cfg["base_model"]["projector_path"], map_location = "cpu")
        encoder_projector.load_state_dict(state_dict, strict = False)
    
    encoder_projector = encoder_projector.to(dtype = dtype, device = device)

    return encoder_projector

# Refer to this https://github.com/Joluck/RWKV-PEFT
# Using Miss in peft
# Typically, papers on LLM-based ASR propose a multi-stage training approach. 
# In the first stage, the LLM is usually frozen while only the projector is trained. 
# In the second and third stages, the LLM is then fine-tuned using LoRA.

def load_rwkv_with_peft(
    args,
    base_ckpt: str,
    adapter_dir: str | None = None,
    dtype = torch.bfloat16,
    device = "cuda",
):
    args = RWKVArgs(**args)
    # 1. base model
    model = RWKV7ModelForCausalLLMCuda(args)
    state_dict = torch.load(base_ckpt, map_location = "cpu", weights_only = True)
    model.load_state_dict(state_dict, strict=False)
    model = model.to(dtype = dtype, device = device)
    

    if args.peft_r != 0:

        # 2. config for PEFT / HF compatibility
        model.config = RWKVConfig(n_embd = args.n_embd, n_layer = args.n_layer)

        # 3. load adapter if provided
        if adapter_dir is not None:
            model = PeftModel.from_pretrained(
                model,
                adapter_dir,
                is_trainable = True,
            )
        else:
            peft_config = MissConfig(
                task_type = TaskType.CAUSAL_LM,
                r = args.peft_r,
                target_modules = ["receptance", "key", "value", "output"]
            )

            model = get_peft_model(model, peft_config)
            model.print_trainable_parameters()

        print(model)

        model.get_base_model().eval()
    else:
        model.eval()

    return model
import os
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

# import deepspeed
from torch.utils.checkpoint import checkpoint as torch_checkpoint
from torch.utils.cpp_extension import load


current_dir = os.path.dirname(os.path.abspath(__file__))

# For training with bf16
HEAD_SIZE = int(os.environ.get("RWKV_HEAD_SIZE_A", 64))
CHUNK_LEN = 16
flags = ['-res-usage', f'-D_N_={HEAD_SIZE}', f"-D_CHUNK_LEN_={CHUNK_LEN}", "--use_fast_math", "-O3", "-Xptxas -O3", "--extra-device-vectorization"]
load(name="rwkv7_clampw", sources=[f'{current_dir}/cuda/rwkv7_clampw.cu', f'{current_dir}/cuda/rwkv7_clampw.cpp'], is_python_module=False, verbose=True, extra_cuda_cflags=flags)
class WindBackstepping(torch.autograd.Function):
    @staticmethod
    def forward(ctx,r,w,k,v,a,b):
        B,T,H,C = r.shape 
        assert T%CHUNK_LEN == 0
        assert all(i.dtype==torch.bfloat16 for i in [r,w,k,v,a,b])
        assert all(i.is_contiguous() for i in [r,w,k,v,a,b])
        y = torch.empty_like(v)
        s = torch.empty(B,H,T//CHUNK_LEN,C,C, dtype=torch.float32,device=w.device)
        sa = torch.empty(B,T,H,C, dtype=torch.float32,device=w.device)
        torch.ops.rwkv7_clampw.forward(r,w,k,v,a,b,y,s,sa)
        ctx.save_for_backward(r,w,k,v,a,b,s,sa)
        return y
    @staticmethod
    def backward(ctx,dy):
        assert all(i.dtype==torch.bfloat16 for i in [dy])
        assert all(i.is_contiguous() for i in [dy])
        r,w,k,v,a,b,s,sa = ctx.saved_tensors
        dr,dw,dk,dv,da,db = [torch.empty_like(x) for x in [r,w,k,v,a,b]]
        torch.ops.rwkv7_clampw.backward(r,w,k,v,a,b,dy,s,sa,dr,dw,dk,dv,da,db)
        return dr,dw,dk,dv,da,db
    
def RUN_CUDA_RWKV7g(r,w,k,v,a,b):
    B, T, HC = r.shape
    r, w, k, v, a, b = [i.view(B, T, HC//64,64).bfloat16().contiguous() for i in [r, w, k, v, a, b]]
    return WindBackstepping.apply(r, w, k, v, a, b).view(B, T, HC)


# For inference with fp16
# TODO: 
# The current code has not been optimized
# the official optimized the https://github.com/BlinkDL/Albatross/blob/main/faster_251101/reference/rwkv7.py.

DTYPE = torch.half
load(name="rwkv7_state_fwd_fp16", sources=[f"{current_dir}/cuda/rwkv7_state_fwd_fp16.cpp", f"{current_dir}/cuda/rwkv7_state_fwd_fp16.cu"], is_python_module=False,
                    verbose=True, extra_cuda_cflags=["-res-usage", "--use_fast_math", "-O3", "--extra-device-vectorization", f"-D_N_={HEAD_SIZE}"] + (["-Xptxas -O3"] if os.name != "nt" else []))


class WKV_7_BATCH(torch.autograd.Function):
    @staticmethod
    def forward(ctx, state, r, w, k, v, a, b, elapsed_t):
        with torch.no_grad():
            B, C = r.size()
            H = C // HEAD_SIZE
            y = torch.empty((B, C), device=k.device, dtype=DTYPE, requires_grad=False, memory_format=torch.contiguous_format)
            torch.ops.rwkv7_state_fwd_fp16.forward_one(B, C, H, state, r, w, k, v, a, b, y, elapsed_t)
            return y, state
        
def RWKV7_ONE_BATCH_OP(state, r, w, k, v, a, b, elapsed_t):
    B, T, C = r.shape
    state = state.half()
    r, w, k, v, a, b = [i.view(B, -1).half().contiguous() for i in [r, w, k, v, a, b]]
    return WKV_7_BATCH.apply(state, r, w, k, v, a, b, elapsed_t)

class WKV_7_SEQ_BATCH(torch.autograd.Function):
    @staticmethod
    def forward(ctx, state, r, w, k, v, a, b, elapsed_t):
        with torch.no_grad():
            B, T, C = r.size()
            H = C // HEAD_SIZE
            y = torch.empty((B, T, C), device=k.device, dtype=DTYPE, requires_grad=False, memory_format=torch.contiguous_format)
            torch.ops.rwkv7_state_fwd_fp16.forward_seq(B, T, C, H, state, r, w, k, v, a, b, y, elapsed_t)
            return y, state


def RWKV7_BATCH_OP(state:torch.Tensor, r:torch.Tensor, w:torch.Tensor, k:torch.Tensor, v:torch.Tensor, a:torch.Tensor, b:torch.Tensor, elapsed_t:torch.Tensor) -> torch.Tensor:
    state = state.half()
    r, w, k, v, a, b = [i.half().contiguous() for i in [r, w, k, v, a, b]]
    return WKV_7_SEQ_BATCH.apply(state, r, w, k, v, a, b, elapsed_t)


class RWKV_Tmix_x070(nn.Module):
    def __init__(self, args, layer_id):
        super().__init__()
        self.args = args
        self.layer_id = layer_id

        self.head_size = args.head_size_a
        self.n_head = args.n_embd // self.head_size
        assert args.n_embd % self.n_head == 0
        H = self.n_head
        N = self.head_size
        C = args.n_embd

        with torch.no_grad():
            ratio_0_to_1 = layer_id / (args.n_layer - 1)  # 0 to 1
            ratio_1_to_almost0 = 1.0 - (layer_id / args.n_layer)  # 1 to ~0
            ddd = torch.ones(1, 1, C)
            for i in range(C):
                ddd[0, 0, i] = i / C

            self.x_r = nn.Parameter(1.0 - torch.pow(ddd, 0.2 * ratio_1_to_almost0))
            self.x_w = nn.Parameter(1.0 - torch.pow(ddd, 0.9 * ratio_1_to_almost0))
            self.x_k = nn.Parameter(1.0 - torch.pow(ddd, 0.7 * ratio_1_to_almost0))
            self.x_v = nn.Parameter(1.0 - torch.pow(ddd, 0.7 * ratio_1_to_almost0))
            self.x_a = nn.Parameter(1.0 - torch.pow(ddd, 0.9 * ratio_1_to_almost0))
            self.x_g = nn.Parameter(1.0 - torch.pow(ddd, 0.2 * ratio_1_to_almost0))

            def ortho_init(x, scale):
                with torch.no_grad():
                    shape = x.shape
                    if len(shape) == 2:
                        gain = math.sqrt(shape[0] / shape[1]) if shape[0] > shape[1] else 1
                        nn.init.orthogonal_(x, gain=gain * scale)
                    elif len(shape) == 3:
                        gain = math.sqrt(shape[1] / shape[2]) if shape[1] > shape[2] else 1
                        for i in range(shape[0]):
                            nn.init.orthogonal_(x[i], gain=gain * scale)
                    else:
                        assert False
                    return x

            www = torch.zeros(C)
            zigzag = torch.zeros(C)
            linear = torch.zeros(C)
            for n in range(C):
                linear[n] = n / (C-1) - 0.5
                zigzag[n] = ((n % N) - ((N-1) / 2)) / ((N-1) / 2)
                zigzag[n] = zigzag[n] * abs(zigzag[n])
                www[n] = -6 + 6 * (n / (C - 1)) ** (1 + 1 * ratio_0_to_1 ** 0.3)

            # D_DECAY_LORA = 64
            D_DECAY_LORA = max(32, int(round(  (1.8*(C**0.5))  /32)*32)) # suggestion
            self.w1 = nn.Parameter(torch.zeros(C, D_DECAY_LORA))
            self.w2 = nn.Parameter(ortho_init(torch.zeros(D_DECAY_LORA, C), 0.1))
            self.w0 = nn.Parameter(www.reshape(1,1,C) + 0.5 + zigzag*2.5) # !!! 0.5 comes from F.softplus !!!

            # D_AAA_LORA = 64
            D_AAA_LORA = max(32, int(round(  (1.8*(C**0.5))  /32)*32)) # suggestion
            self.a1 = nn.Parameter(torch.zeros(C, D_AAA_LORA))
            self.a2 = nn.Parameter(ortho_init(torch.zeros(D_AAA_LORA, C), 0.1))
            self.a0 = nn.Parameter(torch.zeros(1,1,C)-0.19 + zigzag*0.3 + linear*0.4)

            # D_MV_LORA = 32
            if layer_id != 0:
                D_MV_LORA = max(32, int(round(  (1.3*(C**0.5))  /32)*32)) # suggestion
                self.v1 = nn.Parameter(torch.zeros(C, D_MV_LORA))
                self.v2 = nn.Parameter(ortho_init(torch.zeros(D_MV_LORA, C), 0.1))
                self.v0 = nn.Parameter(torch.zeros(1,1,C)+0.73 - linear*0.4)

            # Note: for some data, you can reduce D_GATE_LORA or even remove this gate
            # D_GATE_LORA = 128
            D_GATE_LORA = max(32, int(round(  (0.6*(C**0.8))  /32)*32)) # suggestion
            self.g1 = nn.Parameter(torch.zeros(C, D_GATE_LORA))
            self.g2 = nn.Parameter(ortho_init(torch.zeros(D_GATE_LORA, C), 0.1))

            self.k_k = nn.Parameter(torch.zeros(1,1,C)+0.71 - linear*0.1)
            self.k_a = nn.Parameter(torch.zeros(1,1,C)+1.02)
            self.r_k = nn.Parameter(torch.zeros(H,N)-0.04)

            self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))
            self.receptance = nn.Linear(C, C, bias=False)
            self.key = nn.Linear(C, C, bias=False)
            self.value = nn.Linear(C, C, bias=False)
            self.output = nn.Linear(C, C, bias=False)
            self.ln_x = nn.GroupNorm(H, C, eps=(1e-5)*(args.head_size_divisor**2)) # !!! notice eps value !!!

            # !!! initialize if you are using RWKV_Tmix_x070 in your code !!!
            if args.need_init_tmix:
                self._init_params(args)

    def _init_params(self, args):
        C = args.n_embd
        self.receptance.weight.data.uniform_(-0.5/(C**0.5), 0.5/(C**0.5))
        self.key.weight.data.uniform_(-0.05/(C**0.5), 0.05/(C**0.5))
        self.value.weight.data.uniform_(-0.5/(C**0.5), 0.5/(C**0.5))
        self.output.weight.data.zero_()

    @torch.inference_mode()
    def forward_batch(self, x, attention_mask = None, v_first = None, x_prev = None, state = None, elapsed_t = None):
        B, T, C = x.size()
        x = x.mul(attention_mask)
        H = self.n_head
        xx = torch.cat((x_prev[0].unsqueeze(1), x[:,:-1,:]), dim=1) - x
        x_prev[0] = x[:, -1, :]

        xr = x + xx * self.x_r
        xw = x + xx * self.x_w
        xk = x + xx * self.x_k
        xv = x + xx * self.x_v
        xa = x + xx * self.x_a
        xg = x + xx * self.x_g

        r = self.receptance(xr)
        w = self.w0 + torch.tanh(xw @ self.w1) @ self.w2 # will be soft-clamped to (-inf, -0.5) and exp(-exp(w)) in RWKV7_CLAMPW_CUDA kernel
        k = self.key(xk)
        v = self.value(xv)
        if self.layer_id == 0:
            v_first = v # store the v of the first layer
        else:
            v = v + (v_first - v) * torch.sigmoid(self.v0 + (xv @ self.v1) @ self.v2) # add value residual
        a = torch.sigmoid(self.a0 + (xa @ self.a1) @ self.a2) # a is "in-context learning rate"
        g = torch.sigmoid(xg @ self.g1) @ self.g2

        kk = k * self.k_k
        kk = F.normalize(kk.view(B,T,H,-1), dim=-1, p=2.0).view(B,T,C)
        k = k * (1 + (a-1) * self.k_a)
        v = v * attention_mask
        if T == 1:
            x, state_out = RWKV7_ONE_BATCH_OP(state, r, w, k, v, -kk, kk*a, elapsed_t)
        else:
            x, state_out = RWKV7_BATCH_OP(state,r, w, k, v, -kk, kk*a, elapsed_t)

        x = self.ln_x(x.view(B * T, C)).view(B, T, C)
        x = x + ((r.view(B,T,H,-1)*k.view(B,T,H,-1)*self.r_k).sum(dim=-1, keepdim=True) * v.view(B,T,H,-1)).view(B,T,C)
        x = self.output(x * g)
        return x, v_first, x_prev, state_out

    def forward(self, x,attention_mask = None, v_first=None):
        B, T, C = x.size()
        x = x.mul(attention_mask)
        H = self.n_head
        xx = self.time_shift(x) - x

        xr = x + xx * self.x_r
        xw = x + xx * self.x_w
        xk = x + xx * self.x_k
        xv = x + xx * self.x_v
        xa = x + xx * self.x_a
        xg = x + xx * self.x_g

        r = self.receptance(xr)
        w = self.w0 + torch.tanh(xw @ self.w1) @ self.w2 # will be soft-clamped to (-inf, -0.5) and exp(-exp(w)) in RWKV7_CLAMPW_CUDA kernel
        k = self.key(xk)
        v = self.value(xv)
        if self.layer_id == 0:
            v_first = v # store the v of the first layer
        else:
            v = v + (v_first - v) * torch.sigmoid(self.v0 + (xv @ self.v1) @ self.v2) # add value residual
        a = torch.sigmoid(self.a0 + (xa @ self.a1) @ self.a2) # a is "in-context learning rate"
        g = torch.sigmoid(xg @ self.g1) @ self.g2

        kk = k * self.k_k
        kk = F.normalize(kk.view(B,T,H,-1), dim=-1, p=2.0).view(B,T,C)
        k = k * (1 + (a-1) * self.k_a)
        v = v * attention_mask
        x = RUN_CUDA_RWKV7g(r, w, k, v, -kk, kk*a)
        x = self.ln_x(x.view(B * T, C)).view(B, T, C)

        x = x + ((r.view(B,T,H,-1)*k.view(B,T,H,-1)*self.r_k).sum(dim=-1, keepdim=True) * v.view(B,T,H,-1)).view(B,T,C)
        x = self.output(x * g)
        return x, v_first

class RWKV_CMix_x070(nn.Module):
    def __init__(self, args, layer_id):
        super().__init__()
        self.args = args
        self.layer_id = layer_id
        self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))

        with torch.no_grad():
            ratio_1_to_almost0 = 1.0 - (layer_id / args.n_layer)  # 1 to ~0
            ddd = torch.ones(1, 1, args.n_embd)
            for i in range(args.n_embd):
                ddd[0, 0, i] = i / args.n_embd
            self.x_k = nn.Parameter(1.0 - torch.pow(ddd, ratio_1_to_almost0**4))

        self.key = nn.Linear(args.n_embd, args.n_embd * 4, bias=False)
        self.value = nn.Linear(args.n_embd * 4, args.n_embd, bias=False)

        # !!! initialize if you are using RWKV_Tmix_x070 in your code !!!
        if args.need_init_cmix:
            self._init_params(args)

    def _init_params(self, args):
        self.key.weight.data.uniform_(-0.5/(args.n_embd**0.5), 0.5/(args.n_embd**0.5))
        self.value.weight.data.zero_()

    @torch.inference_mode()
    def forward_batch(self, x,attention_mask = None, x_prev=None):
        B, T, C = x.size()
        x = x.mul(attention_mask)
        xx = torch.cat((x_prev[1].unsqueeze(1), x[:,:-1,:]), dim=1) - x
        x_prev[1] = x[:, -1, :]
        k = x + xx * self.x_k
        k = torch.relu(self.key(k)) ** 2

        return self.value(k), x_prev

    def forward(self, x,attention_mask):
        x = x.mul(attention_mask)
        xx = self.time_shift(x) - x
        
        k = x + xx * self.x_k
        k = torch.relu(self.key(k)) ** 2

        return self.value(k)
    
class Block(nn.Module):
    def __init__(self, args, layer_id, attn=None):
        super().__init__()
        self.args = args
        self.layer_id = layer_id

        self.ln1 = nn.LayerNorm(args.n_embd)
        self.ln2 = nn.LayerNorm(args.n_embd)

        if self.layer_id == 0:
            self.ln0 = nn.LayerNorm(args.n_embd)

        if attn is None:
            self.att = RWKV_Tmix_x070(args, layer_id)
        else:
            self.att = attn
        self.ffn = RWKV_CMix_x070(args, layer_id)
        
        if args.dropout > 0:
            self.drop0 = nn.Dropout(p = args.dropout)
            self.drop1 = nn.Dropout(p = args.dropout)

    @torch.inference_mode()
    def forward_batch(self, x,attention_mask = None, v_first=None, x_prev=None,state=None, elapsed_t=None):
        if self.layer_id == 0:
            x = self.ln0(x)
        x_attn, v_first, x_prev, state = self.att.forward_batch(self.ln1(x), attention_mask, v_first, x_prev, state, elapsed_t)
        x = x + x_attn
        x_ffn, x_prev = self.ffn.forward_batch(self.ln2(x), attention_mask, x_prev)
        x = x + x_ffn
        return x, v_first, x_prev, state

    def forward(self, x,attention_mask, v_first=None):
        if self.layer_id == 0:
            x = self.ln0(x)

        x_attn, v_first = self.att(self.ln1(x), attention_mask, v_first)
        x = x + x_attn

        x = x + self.ffn(self.ln2(x),attention_mask)
        return x, v_first
    
class L2Wrap(torch.autograd.Function):
    def forward(ctx, loss, y):
        ctx.save_for_backward(y)
        return loss

    def backward(ctx, grad_output):
        y = ctx.saved_tensors[0]
        # to encourage the logits to be close to 0
        factor = 1e-4 / (y.shape[0] * y.shape[1])
        maxx, ids = torch.max(y, -1, keepdim=True)
        gy = torch.zeros_like(y)
        gy.scatter_(-1, ids, maxx * factor)
        return (grad_output, gy)
    
class RWKV7ModelForLatentInputsCuda(nn.Module):
    def __init__(self, args=None, **kwargs):
        super().__init__()
        self.args = args
        assert args.n_embd % 32 == 0


        self.blocks = nn.ModuleList([Block(args, i) for i in range(args.n_layer)])
        self.ln_out = nn.LayerNorm(args.n_embd)
        if args.dropout > 0:
            self.drop0 = nn.Dropout(p = args.dropout)
            
    def forward(self, latents, attention_mask = None):
        args = self.args
        B, T, C = latents.size()
    
        if attention_mask is None:
            attention_mask = torch.ones(B, T, dtype=torch.bool, device=latents.device)
        else:
            assert attention_mask.shape == (B, T), f'attention_mask shape: {attention_mask.shape}, latents shape: {latents.shape}'
        
        if T % 16 != 0:
            # right padding to 16x
            padding_length = 16 - T % 16
            latents = torch.cat([latents, torch.zeros(B, padding_length, C,device=latents.device,dtype=latents.dtype)], dim=1)
            attention_mask = torch.cat([attention_mask, torch.zeros(B, padding_length, dtype=torch.bool, device=latents.device)], dim=1)

        attention_mask = attention_mask.unsqueeze(-1)
        x = latents
        if args.dropout > 0:
            x = self.drop0(x)
        v_first = torch.empty_like(x)
        for i in range(args.n_layer):
            block = self.blocks[i]
            if self.training and args.grad_cp == 1:
                # NOTE: 
                # Currently, grad_cp still has bugs, mainly related to data type issues, with bf16 or fp16, 
                # and some operations may become fp32, which can cause problems with deepseed. 
                # However, since it is frozen LLM, the difference is not too much.
                x, v_first = torch_checkpoint(block, x, attention_mask, v_first, use_reentrant=False)
            else:
                x, v_first = block(x,attention_mask, v_first)
        x = self.ln_out(x)
        x = x[:, :T]
        return x



class RWKV7ModelForCausalLLMCuda(nn.Module):
    def __init__(self, args=None, **kwargs):
        super().__init__()

        self.args = args
        assert args.n_embd % 32 == 0

        self.emb = nn.Embedding(args.vocab_size, args.n_embd)
        self.blocks = nn.ModuleList([Block(args, i) for i in range(args.n_layer)])
        self.ln_out = nn.LayerNorm(args.n_embd)
        self.head = nn.Linear(args.n_embd, args.vocab_size, bias=False)
        if args.dropout > 0:
            self.drop0 = nn.Dropout(p = args.dropout)

    # copy from https://github1s.com/Joluck/RWKV-PEFT/blob/HEAD/rwkvt/rwkv7/model.py#L25-L30
    # for peft
    def prepare_inputs_for_generation(self, embds, **kwargs):
        """
        兼容 transformers 的 generate() 接口.
        对 RWKV 来说，我们不需要做实际处理，直接返回原始输入即可。
        """

        return {"embds": embds, **kwargs}

    @torch.inference_mode()
    def forward_batch(self, embds, attention_mask = None, states = None):
        args = self.args
        B, T, C = embds.size()
        if states is None:
            states = [None for _ in range(args.n_layer * 3)]
            for i in range(args.n_layer): # state: 0=att_x_prev 1=att_kv 2=ffn_x_prev
                states[i * 3 + 0] = torch.zeros((2, B, args.n_embd), dtype=DTYPE, requires_grad=False, device="cuda")
                states[i * 3 + 1] = torch.zeros((B, args.n_embd // args.head_size_a, args.head_size_a, args.head_size_a), dtype=torch.float, requires_grad=False, device="cuda")
                states[i * 3 + 2] = torch.zeros((B,) , dtype=torch.int32, requires_grad=False, device="cuda")
        if attention_mask is None:
            attention_mask = torch.ones(B, T, dtype=torch.bool, device=embds.device)
        else:
            assert attention_mask.shape == (B, T), f'attention_mask shape: {attention_mask.shape}, embds shape: {embds.shape}'
        # if T % 16 != 0:
        #     #right padding to 16x
        #     padding_length = 16 - T % 16
        #     embds = torch.cat([torch.zeros(B, padding_length, C,device=embds.device,dtype=embds.dtype), embds], dim=1)
        #     attention_mask = torch.cat([torch.zeros(B, padding_length, dtype=torch.bool, device=embds.device), attention_mask], dim=1)
        attention_mask = attention_mask.unsqueeze(-1)
        x = embds
        v_first = torch.empty_like(x)
        for i in range(args.n_layer):
            block = self.blocks[i]
            x, v_first, x_prev, state = block.forward_batch(
                x, 
                attention_mask, 
                v_first,
                x_prev = states[i * 3 + 0],
                state = states[i * 3 + 1], 
                elapsed_t = states[i * 3 + 2],
            )
            states[i * 3 + 0] = x_prev
            states[i * 3 + 1] = state
            states[i * 3 + 2] += T
        x = x[:, -1,:]
        x = self.ln_out(x)
        logits = self.head(x)
        return x, logits, states
    
    def forward(self, embds = None, attention_mask = None, v_first = None, **kwargs):
        args = self.args
        # embds = self.emb(embds)
        if "embds" in kwargs:
            embds = kwargs["embds"]
        if "input_ids" in kwargs:
            embds = kwargs["input_ids"]
        B, T, C = embds.size()

        if attention_mask is None:
            attention_mask = torch.ones(B, T, dtype=torch.bool, device=embds.device)
        else:
            assert attention_mask.shape == (B, T), f'attention_mask shape: {attention_mask.shape}, embds shape: {embds.shape}'
        if T % 16 != 0:
            # right padding to 16x
            padding_length = 16 - T % 16
            embds = torch.cat([torch.zeros(B, padding_length, C,device=embds.device,dtype=embds.dtype), embds], dim=1)
            attention_mask = torch.cat([torch.zeros(B, padding_length, dtype=torch.bool, device=embds.device), attention_mask], dim=1)
        attention_mask = attention_mask.unsqueeze(-1)
        x = embds
        if args.dropout > 0:
            x = self.drop0(x)
        v_first = torch.empty_like(x)
        for i in range(args.n_layer):
            block = self.blocks[i]
            if self.training and args.grad_cp == 1:
                # NOTE: 
                # Currently, grad_cp still has bugs, mainly related to data type issues, with bf16 or fp16, 
                # and some operations may become fp32, which can cause problems with deepseed. 
                # However, since it is frozen LLM, the difference is not too much.
                # x, v_first = deepspeed.checkpointing.checkpoint(block, x, attention_mask, v_first)
                x, v_first = torch_checkpoint(block, x, attention_mask, v_first, use_reentrant=False)
            else:
                x, v_first = block(x,attention_mask, v_first)
        x = self.ln_out(x)
        x = self.head(x)
        x = x[:, -T:]
        return x
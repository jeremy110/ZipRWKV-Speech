import torch
from torch import nn

from models.zipformer.scaling import SwooshR

from models.utils import RWKVArgs
from models.rwkv.rwkv7 import RWKV7ModelForLatentInputsCuda

# https://github.com/zhu-han/SpeechLLM/blob/master/src/model.py#L467
class EncoderProjector(nn.Module):
    """
    The encoder projector module. It is used to project the encoder outputs to the same dimension as the language model.
    Modified from https://github.com/X-LANCE/SLAM-LLM/blob/main/src/slam_llm/models/projector.py.
    Args:
        encoder_dim (:obj:`int`): The dimension of the encoder outputs.
        llm_dim (:obj:`int`): The dimension of the language model.
        downsample_rate (:obj:`int`, `optional`, defaults to 2): The downsample rate to use.
    """

    def __init__(self, encoder_dim, llm_dim, downsample_rate=2):
        super().__init__()
        self.downsample_rate = downsample_rate
        self.proj = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(encoder_dim * self.downsample_rate, llm_dim),
            SwooshR(),
            # nn.ReLU(),
            nn.Linear(llm_dim, llm_dim),
        )

    def forward(self, x):

        batch_size, seq_len, feat_dim = x.size()
        num_frames_to_discard = seq_len % self.downsample_rate
        if num_frames_to_discard > 0:
            x = x[:, :-num_frames_to_discard, :]
        seq_len = x.size(1)

        x = x.contiguous()
        x = x.view(
            batch_size, seq_len // self.downsample_rate, feat_dim * self.downsample_rate
        )

        x = self.proj(x)

        return x
    

    
class EncoderProjector_rwkv(nn.Module):
    """
        https://github.com/yynil/RWKVTTS/blob/main/model/llm/rwkv_asr_cuda_whisper.py#L515
        https://huggingface.co/yueyulin/rwkv_asr
    """

    def __init__(self, encoder_dim, llm_dim, downsample_rate = 2, llm_args = None):
        super().__init__()
        args = RWKVArgs(**llm_args)
        self.k = downsample_rate
        self.proj_down = nn.Linear(encoder_dim * self.k, args.n_embd)
        self.audio_lm_model = RWKV7ModelForLatentInputsCuda(args)
        self.proj_up = nn.Linear(args.n_embd, llm_dim)

    def forward(self, x: torch.Tensor, lens: torch.Tensor):
        batch_size, seq_len, feat_dim = x.size()

        num_frames_to_discard = seq_len % self.k

        if num_frames_to_discard > 0:
            x = x[:, :-num_frames_to_discard, :]
            lens = torch.clamp(lens - num_frames_to_discard, min=0)

        seq_len = x.size(1)

        x = x.contiguous()
        x = x.view(batch_size, seq_len // self.k, feat_dim * self.k)
        lens = lens // self.k

        attention_mask = torch.arange(x.size(1), device=x.device).unsqueeze(0) < lens.unsqueeze(1)

        x = self.proj_down(x)
        x = self.audio_lm_model(x, attention_mask)
        x = self.proj_up(x)

        return x, lens
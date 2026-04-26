import torch
from torch.utils.cpp_extension import load
import torch.nn.functional as F

flash_module = load(
    name="flash_attn_v1",
    sources=["./flash_fwd_sm80.cu"],
    extra_cuda_cflags=["-arch=sm_89"],
)

# Phase 1 test - small, fixed, reproducible
torch.manual_seed(42)
batch, seqlen, nheads, hdim = 1, 64, 1, 64
dtype = torch.float16

q = torch.randn(batch, seqlen, nheads, hdim, device="cuda", dtype=dtype)
k = torch.randn(batch, seqlen, nheads, hdim, device="cuda", dtype=dtype)
v = torch.randn(batch, seqlen, nheads, hdim, device="cuda", dtype=dtype)

# Reference (note: SDPA expects (batch, nheads, seqlen, hdim))
q_t = q.transpose(1, 2)
k_t = k.transpose(1, 2)
v_t = v.transpose(1, 2)
ref = F.scaled_dot_product_attention(q_t, k_t, v_t).transpose(1, 2)

# Your kernel
out = flash_module.forward(q, k, v)

print(f"max diff: {(ref - out).abs().max().item()}")

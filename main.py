import torch
from torch.utils.cpp_extension import load
import torch.nn.functional as F

flash_module = load(
    name="flash_attn_v1",
    sources=["./flash_fwd_sm80.cu"],
    extra_cuda_cflags=["-arch=sm_89"],
    extra_include_paths=["/usr/include/python3.11"],
)


# Phase 1 test - small, fixed, reproducible
torch.manual_seed(42)
batch, seqlen, nheads, hdim = 1, 64, 1, 64
dtype = torch.float16

# Test softmax - needs 2D tensor, cols must be 128/256/512 (n/32 = 4/8/16)
x = torch.randn(4, 128, device="cuda", dtype=torch.float32)
ref_softmax = torch.softmax(x, dim=-1)
out_softmax = flash_module.softmax(x)
print(f"softmax max diff: {(ref_softmax - out_softmax).abs().max().item()}")

q = torch.randn(batch, seqlen, nheads, hdim, device="cuda", dtype=dtype)
k = torch.randn(batch, seqlen, nheads, hdim, device="cuda", dtype=dtype)
v = torch.randn(batch, seqlen, nheads, hdim, device="cuda", dtype=dtype)

# Reference (note: SDPA expects (batch, nheads, seqlen, hdim))
q_t = q.transpose(1, 2)
k_t = k.transpose(1, 2)
v_t = v.transpose(1, 2)
ref = F.scaled_dot_product_attention(q_t, k_t, v_t).transpose(1, 2)

out = flash_module.vanilla_attention(q.float(), k.float(), v.float())
ref_f32 = F.scaled_dot_product_attention(q_t.float(), k_t.float(), v_t.float()).transpose(1, 2)
print(f"vanilla_attention max diff: {(ref_f32 - out).abs().max().item()}")

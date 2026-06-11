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
batch, seqlen, nheads, hdim = 1, 128, 1, 128
dtype = torch.float16


q = torch.randn(batch, seqlen, nheads, hdim, device="cuda", dtype=dtype)
k = torch.randn(batch, seqlen, nheads, hdim, device="cuda", dtype=dtype)
v = torch.randn(batch, seqlen, nheads, hdim, device="cuda", dtype=dtype)

# Reference (note: SDPA expects (batch, nheads, seqlen, hdim))
q_t = q.transpose(1, 2)
k_t = k.transpose(1, 2)
v_t = v.transpose(1, 2)

ref_f32 = F.scaled_dot_product_attention(
    q_t.float(), k_t.float(), v_t.float()
).transpose(1, 2)

# Warmup
for _ in range(500):
    flash_module.vanilla_attention(q.float(), k.float(), v.float())
    flash_module.flash_attention_v1(q.float(), k.float(), v.float())
    flash_module.flash_attention_v2(q.float(), k.float(), v.float())
torch.cuda.synchronize()

start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)

start.record()
out = flash_module.vanilla_attention(q.float(), k.float(), v.float())
end.record()
torch.cuda.synchronize()
print(f"vanilla_attention:        {start.elapsed_time(end):.3f} ms")

start.record()
flash_attention_out = flash_module.flash_attention_v1(q.float(), k.float(), v.float())
end.record()
torch.cuda.synchronize()
print(f"flash_attention_v1:       {start.elapsed_time(end):.3f} ms")

start.record()
flash_attention_out_2 = flash_module.flash_attention_v2(q.float(), k.float(), v.float())
end.record()
torch.cuda.synchronize()
print(f"flash_attention_v2:       {start.elapsed_time(end):.3f} ms")

print(f"vanilla_attention max diff:  {(ref_f32 - out).abs().max().item()}")
print(f"flash_attention_v1 max diff: {(ref_f32 - flash_attention_out).abs().max().item()}")
print(f"flash_attention_v1 max diff: {(ref_f32 - flash_attention_out_2).abs().max().item()}")

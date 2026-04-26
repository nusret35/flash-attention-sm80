import torch
from torch.utils.cpp_extension import load

flash_module = load(
    name="flash_attn_v1",
    sources=["agent_space/flash_fwd_v1_sm80.cu"],
    extra_cuda_cflags=["-arch=sm_89"],
)

torch.manual_seed(42)
rows, cols = 4, 128
x = torch.randn(rows, cols, device="cuda", dtype=torch.float16)

ref = torch.softmax(x, dim=-1)
out = flash_module.softmax(x)

print(f"max diff: {(ref - out).abs().max().item()}")

#include <torch/extension.h>
#include "util.cuh"
#define TILE 32

__global__ void flash_fwd_kernel() {}

__global__ void matmul_tiled(const float *A, const float *B, float *C,
                             int M, int N, int K)
{
  __shared__ float sh_A[TILE][TILE];
  __shared__ float sh_B[TILE][TILE];

  int row = blockIdx.y * TILE + threadIdx.y;
  int col = blockIdx.x * TILE + threadIdx.x;
  float sum = 0.0f;

  for (int t = 0; t < (K + TILE - 1) / TILE; t++)
  {
    if ((row < M) && ((t * TILE + threadIdx.x) < N))
      sh_A[threadIdx.y][threadIdx.x] = A[row * K + t * TILE + threadIdx.x];
    else
      sh_A[threadIdx.y][threadIdx.x] = 0.0f;

    if (((t * TILE + threadIdx.y) < N) && (col < K))
      sh_B[threadIdx.y][threadIdx.x] = B[(t * TILE + threadIdx.y) * N + col];
    else
      sh_B[threadIdx.y][threadIdx.x] = 0.0f;

    __syncthreads();

    for (int i = 0; i < TILE; i++)
      sum += sh_A[threadIdx.y][i] * sh_B[i][threadIdx.x];
    __syncthreads();
  }

  if ((row < M) && (col < K))
    C[row * N + col] = sum;
}

torch::Tensor softmax_forward(torch::Tensor input)
{
  auto sizes = input.sizes();
  int m = sizes[0];
  int n = sizes[1];

  auto output = torch::empty_like(input);
  dim3 block(32, 1);
  dim3 grid(m);

  auto *in_ptr = (const float4 *)input.data_ptr<float>();
  auto *out_ptr = (float4 *)output.data_ptr<float>();

  int cpt = n / 32;
  if (cpt == 4)
  {
    softmax_stored_locally_mutli_dim<4><<<grid, block>>>(in_ptr, out_ptr, m, n);
  }
  else if (cpt == 8)
  {
    softmax_stored_locally_mutli_dim<8><<<grid, block>>>(in_ptr, out_ptr, m, n);
  }
  else if (cpt == 16)
  {
    softmax_stored_locally_mutli_dim<16>
        <<<grid, block>>>(in_ptr, out_ptr, m, n);
  }

  return output;
}

torch::Tensor vanilla_attention(torch::Tensor q, torch::Tensor k, torch::Tensor v)
{
  // q, k, v: (batch, seqlen, nheads, hdim) - contiguous float32
  int seqlen = q.size(1);
  int hdim = q.size(3);

  // Step 1: S = Q @ K^T  → (seqlen × seqlen)
  auto scores = torch::empty({seqlen, seqlen}, q.options());
  auto kt = k.transpose(1, 3).contiguous(); // (batch, hdim, nheads, seqlen) → treat as (hdim × seqlen)

  dim3 block(TILE, TILE);
  dim3 grid1((seqlen + TILE - 1) / TILE, (seqlen + TILE - 1) / TILE);
  matmul_tiled<<<grid1, block>>>(
      q.data_ptr<float>(), kt.data_ptr<float>(), scores.data_ptr<float>(),
      seqlen, seqlen, hdim);

  // Step 2: scale by 1/sqrt(d)
  scores.div_(sqrtf(hdim));

  // Step 3: attn_weights = softmax(scores)  → (seqlen × seqlen)
  auto attn_weights = softmax_forward(scores);

  // Step 4: O = attn_weights @ V  → (seqlen × hdim)
  auto output = torch::empty_like(q);
  dim3 grid2((hdim + TILE - 1) / TILE, (seqlen + TILE - 1) / TILE);
  matmul_tiled<<<grid2, block>>>(
      attn_weights.data_ptr<float>(), v.data_ptr<float>(), output.data_ptr<float>(),
      seqlen, hdim, seqlen);

  return output;
}

torch::Tensor forward(torch::Tensor q, torch::Tensor k, torch::Tensor v)
{
  auto output = torch::zeros_like(q);
  // TODO: implement flash_fwd_kernel and launch it here
  return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
  m.def("forward", &forward, "Flash attention forward");
  m.def("vanilla_attention", &vanilla_attention, "Vanilla attention forward");
  m.def("softmax", &softmax_forward, "Softmax forward");
}

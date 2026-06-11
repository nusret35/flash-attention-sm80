#include <torch/extension.h>
#include "util.cuh"
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#define TILE 32

#ifndef MAX_HDIM
#define MAX_HDIM 128 // per-thread register arrays sized to this
#endif

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
  if (cpt == 2)
  {
    softmax_stored_locally_mutli_dim<2><<<grid, block>>>(in_ptr, out_ptr, m, n);
  }
  else if (cpt == 4)
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

// Flash Attention v1 kernel: one thread per query row, sequential K/V loop.
// Key difference from v2: O is NORMALIZED after every tile (divide by l_new
// each step), matching the original Flash Attention 1 algorithm.
template <int Br, int Bc>
__global__ void flash_attention_v1_kernel(
    const float *__restrict__ Q, // (B*H, N, d)
    const float *__restrict__ K,
    const float *__restrict__ V,
    float *__restrict__ O,
    int N, int d, float scale)
{
  const int bh = blockIdx.y;
  const int qtile = blockIdx.x;
  const int q_row = qtile * Br + threadIdx.x;

  const float *Qbh = Q + (long)bh * N * d;
  const float *Kbh = K + (long)bh * N * d;
  const float *Vbh = V + (long)bh * N * d;
  float *Obh = O + (long)bh * N * d;

  extern __shared__ float smem[];
  float *Ks = smem;
  float *Vs = smem + Bc * d;

  // Per-row running state in registers
  float q_reg[MAX_HDIM];
  float acc[MAX_HDIM]; // NORMALIZED output accumulator (FA1 style)
  float m_i = -INFINITY;
  float l_i = 0.0f;

  const bool active = (q_row < N);
  if (active)
  {
    for (int x = 0; x < d; ++x)
    {
      q_reg[x] = Qbh[(long)q_row * d + x];
      acc[x] = 0.0f;
    }
  }

  const int Tc = (N + Bc - 1) / Bc;

  for (int j = 0; j < Tc; ++j)
  {
    const int kv_start = j * Bc;
    const int kv_len = (N - kv_start) < Bc ? (N - kv_start) : Bc;

    // Load K_j, V_j into shared memory
    for (int idx = threadIdx.x; idx < kv_len * d; idx += blockDim.x)
    {
      Ks[idx] = Kbh[(long)(kv_start)*d + idx];
      Vs[idx] = Vbh[(long)(kv_start)*d + idx];
    }
    __syncthreads();

    if (active)
    {
      // Compute scores and block-local max
      float s[Bc];
      float m_blk = -INFINITY;
      for (int c = 0; c < kv_len; ++c)
      {
        float dot = 0.0f;
        const float *krow = Ks + (long)c * d;
        for (int x = 0; x < d; ++x)
          dot += q_reg[x] * krow[x];
        dot *= scale;
        s[c] = dot;
        m_blk = fmaxf(m_blk, dot);
      }

      // Exp and block-local sum
      float l_blk = 0.0f;
      for (int c = 0; c < kv_len; ++c)
      {
        s[c] = __expf(s[c] - m_blk);
        l_blk += s[c];
      }

      // FA1: merge stats and normalize immediately
      float m_new = fmaxf(m_i, m_blk);
      float alpha = __expf(m_i - m_new);  // rescale old
      float beta = __expf(m_blk - m_new); // rescale new block
      float l_new = alpha * l_i + beta * l_blk;
      float inv_l_new = 1.0f / l_new;

      // P @ V for this block
      float pv[MAX_HDIM];
      for (int x = 0; x < d; ++x)
        pv[x] = 0.0f;
      for (int c = 0; c < kv_len; ++c)
      {
        float p = s[c]; // already exp(s - m_blk)
        const float *vrow = Vs + (long)c * d;
        for (int x = 0; x < d; ++x)
          pv[x] += p * vrow[x];
      }

      // O_new = (alpha * l_old * O_old + beta * PV) / l_new
      for (int x = 0; x < d; ++x)
        acc[x] = (alpha * l_i * acc[x] + beta * pv[x]) * inv_l_new;

      m_i = m_new;
      l_i = l_new;
    }
    __syncthreads();
  }

  // Output is already normalized — just write it
  if (active)
  {
    for (int x = 0; x < d; ++x)
      Obh[(long)q_row * d + x] = acc[x];
  }
}

torch::Tensor flash_attention_v1(torch::Tensor q, torch::Tensor k, torch::Tensor v)
{
  TORCH_CHECK(q.is_cuda() && k.is_cuda() && v.is_cuda(), "inputs must be CUDA");
  TORCH_CHECK(q.scalar_type() == torch::kFloat32, "float32 only");

  auto qc = q.contiguous(), kc = k.contiguous(), vc = v.contiguous();
  int B = qc.size(0), H = qc.size(1), N = qc.size(2), d = qc.size(3);
  TORCH_CHECK(d <= MAX_HDIM, "hdim exceeds MAX_HDIM");

  float scale = 1.0f / sqrtf((float)d);
  auto O = torch::empty_like(qc);

  constexpr int Br = 64;
  constexpr int Bc = 32;
  int Tr = (N + Br - 1) / Br;

  dim3 grid(Tr, B * H);
  dim3 block(Br);
  size_t smem = (size_t)2 * Bc * d * sizeof(float);

  auto stream = at::cuda::getCurrentCUDAStream();
  flash_attention_v1_kernel<Br, Bc><<<grid, block, smem, stream>>>(
      qc.data_ptr<float>(), kc.data_ptr<float>(), vc.data_ptr<float>(),
      O.data_ptr<float>(), N, d, scale);

  return O;
}

// One thread block = one query tile (Br rows) for one (batch*head).
// Grid: (Tr query-tiles, B*H). This is the FA2 launch: Q tiles run in
// parallel across the sequence, K/V is the inner sequential loop.
template <int Br, int Bc>
__global__ void flash_attention_v2_kernel(
    const float *__restrict__ Q, // (B*H, N, d), contiguous
    const float *__restrict__ K,
    const float *__restrict__ V,
    float *__restrict__ O,
    int N, int d, float scale)
{
  const int bh = blockIdx.y;
  const int qtile = blockIdx.x;
  const int q_row = qtile * Br + threadIdx.x; // this thread's query row

  const float *Qbh = Q + (long)bh * N * d;
  const float *Kbh = K + (long)bh * N * d;
  const float *Vbh = V + (long)bh * N * d;
  float *Obh = O + (long)bh * N * d;

  // K and V tiles live in shared memory: each Bc x d.
  extern __shared__ float smem[];
  float *Ks = smem;
  float *Vs = smem + Bc * d;

  // Per-row running state, kept in registers. O accumulator stays
  // UNNORMALIZED the whole time (the FA2 trick) — we divide by l once
  // at the very end instead of every tile like FA1 does.
  float q_reg[MAX_HDIM];
  float acc[MAX_HDIM];
  float m_i = -INFINITY; // running max
  float l_i = 0.0f;      // running denominator

  const bool active = (q_row < N);
  if (active)
  {
    for (int x = 0; x < d; ++x)
    {
      q_reg[x] = Qbh[(long)q_row * d + x];
      acc[x] = 0.0f;
    }
  }

  const int Tc = (N + Bc - 1) / Bc;

  for (int j = 0; j < Tc; ++j)
  {
    const int kv_start = j * Bc;
    const int kv_len = (N - kv_start) < Bc ? (N - kv_start) : Bc;

    // Cooperatively stage K_j, V_j into shared memory (all threads help,
    // including inactive ones, so the barrier below is uniform).
    for (int idx = threadIdx.x; idx < kv_len * d; idx += blockDim.x)
    {
      Ks[idx] = Kbh[(long)(kv_start)*d + idx];
      Vs[idx] = Vbh[(long)(kv_start)*d + idx];
    }
    __syncthreads();

    if (active)
    {
      // Scores for this tile + this tile's max (one pass).
      float s[Bc];
      float m_blk = -INFINITY;
      for (int c = 0; c < kv_len; ++c)
      {
        float dot = 0.0f;
        const float *krow = Ks + (long)c * d;
        for (int x = 0; x < d; ++x)
          dot += q_reg[x] * krow[x];
        dot *= scale;
        s[c] = dot;
        m_blk = fmaxf(m_blk, dot);
      }

      // Merge tile stats into running stats. ONE rescale per tile.
      float m_new = fmaxf(m_i, m_blk);
      float correction = __expf(m_i - m_new); // rescales old state
      l_i *= correction;
      for (int x = 0; x < d; ++x)
        acc[x] *= correction;

      // Accumulate this tile (no division here — that's the point).
      for (int c = 0; c < kv_len; ++c)
      {
        float p = __expf(s[c] - m_new);
        l_i += p;
        const float *vrow = Vs + (long)c * d;
        for (int x = 0; x < d; ++x)
          acc[x] += p * vrow[x];
      }
      m_i = m_new;
    }
    __syncthreads(); // done reading smem before next tile overwrites it
  }

  // Deferred normalization: single divide, once.
  if (active)
  {
    float inv_l = 1.0f / l_i;
    for (int x = 0; x < d; ++x)
      Obh[(long)q_row * d + x] = acc[x] * inv_l;
  }
}

// Expects (B, H, N, d), float32, CUDA. Generalizes your single-head
// reference to real batch/head dims via the grid's y-axis.
torch::Tensor flash_attention_v2(torch::Tensor q, torch::Tensor k, torch::Tensor v)
{
  TORCH_CHECK(q.is_cuda() && k.is_cuda() && v.is_cuda(), "inputs must be CUDA");
  TORCH_CHECK(q.scalar_type() == torch::kFloat32, "float32 only");

  auto qc = q.contiguous(), kc = k.contiguous(), vc = v.contiguous();
  int B = qc.size(0), H = qc.size(1), N = qc.size(2), d = qc.size(3);
  TORCH_CHECK(d <= MAX_HDIM, "hdim exceeds MAX_HDIM");

  float scale = 1.0f / sqrtf((float)d);
  auto O = torch::empty_like(qc);

  constexpr int Br = 64;
  constexpr int Bc = 32;
  int Tr = (N + Br - 1) / Br;

  dim3 grid(Tr, B * H);
  dim3 block(Br);
  size_t smem = (size_t)2 * Bc * d * sizeof(float);

  auto stream = at::cuda::getCurrentCUDAStream();
  flash_attention_v2_kernel<Br, Bc><<<grid, block, smem, stream>>>(
      qc.data_ptr<float>(), kc.data_ptr<float>(), vc.data_ptr<float>(),
      O.data_ptr<float>(), N, d, scale);

  return O; // (B, H, N, d)
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
  m.def("flash_attention_v2", &flash_attention_v2, "Flash attention 2 forward");
  m.def("flash_attention_v1", &flash_attention_v1, "Flash attention 1 forward");
  m.def("vanilla_attention", &vanilla_attention, "Vanilla attention forward");
  m.def("softmax", &softmax_forward, "Softmax forward");
}

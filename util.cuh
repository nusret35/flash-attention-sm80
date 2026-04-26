#pragma once

#include <cuda_fp16.h>

struct half4 {
  __half x, y, z, w;
};

template <typename T, int NUM>
__inline__ __device__ T warpReduceMax(T *val, int thread_group_width = 32) {
#pragma unroll
  for (int i = 0; i < NUM; i++) {
#pragma unroll
    for (int mask = thread_group_width / 2; mask > 0; mask >>= 1) {
      val[i] = max(val[i], __shfl_xor_sync(0xffffffff, val[i], mask, 32));
    }
  }
  return (T)(0.0f);
}

template <typename T, int NUM>
__inline__ __device__ T warpReduceSum(T *val, int thread_group_width = 32) {
#pragma unroll
  for (int i = 0; i < NUM; i++) {
#pragma unroll
    for (int mask = thread_group_width / 2; mask > 0; mask >>= 1) {
      val[i] += __shfl_xor_sync(0xffffffff, val[i], mask, 32);
    }
  }
  return (T)(0.0f);
}

template <int cols_per_thread>
__global__ void softmax_stored_locally_mutli_dim(const half4 *input,
                                                 half4 *output, size_t m,
                                                 size_t n) {
  constexpr int num_packs =
      (cols_per_thread + 3) /
      4; // pack_size = 4,k/32 = cols_per_thread, num_packs = k/32/4

  float4 buf[num_packs];
  const int m_idx = blockIdx.x * blockDim.y + threadIdx.y;
  const int tid = threadIdx.x;

  for (int64_t row = m_idx; row < m; row += gridDim.x * blockDim.y) {

    const int64_t row_offset = row * (n >> 2);
    const half4 *row_x = input + row_offset;
    half4 *row_y = output + row_offset;
    float local_max[1] = {-INFINITY};
#pragma unroll
    for (int pack_id = 0; pack_id < num_packs; ++pack_id) {
      const int col = pack_id * blockDim.x + tid;
      if (col < n / 4) {
        buf[pack_id] = {
            __half2float(row_x[col].x),
            __half2float(row_x[col].y),
            __half2float(row_x[col].z),
            __half2float(row_x[col].w),
        };
      } else {
        buf[pack_id].x = -INFINITY;
        buf[pack_id].y = -INFINITY;
        buf[pack_id].z = -INFINITY;
        buf[pack_id].w = -INFINITY;
      }
    }
#pragma unroll
    for (int i = 0; i < num_packs; i++) {
      local_max[0] = max(local_max[0], buf[i].x);
      local_max[0] = max(local_max[0], buf[i].y);
      local_max[0] = max(local_max[0], buf[i].z);
      local_max[0] = max(local_max[0], buf[i].w);
    }
    warpReduceMax<float, 1>(local_max, blockDim.x);

    float local_sum[1] = {0.0f};
#pragma unroll
    for (int i = 0; i < num_packs; i++) {
      buf[i].x = exp(buf[i].x - local_max[0]);
      buf[i].y = exp(buf[i].y - local_max[0]);
      buf[i].z = exp(buf[i].z - local_max[0]);
      buf[i].w = exp(buf[i].w - local_max[0]);
      local_sum[0] += buf[i].x;
      local_sum[0] += buf[i].y;
      local_sum[0] += buf[i].z;
      local_sum[0] += buf[i].w;
    }

    warpReduceSum<float, 1>(local_sum, blockDim.x);

    for (int i = 0; i < num_packs; ++i) {
      const int col = i * blockDim.x + tid;
      if (col < n / 4) {
        row_y[col] = {__float2half(buf[i].x / local_sum[0]),
                      __float2half(buf[i].y / local_sum[0]),
                      __float2half(buf[i].z / local_sum[0]),
                      __float2half(buf[i].w / local_sum[0])};
      }
    }
  }
}

#include "util.cuh"
#include <torch/extension.h>

__global__ void flash_fwd_kernel() {}

__global__ void vanilla_att_fwd_kernel(torch::Tensor q, torch::Tensor k,
                                       torch::Tensor v) {}

torch::Tensor softmax_forward(torch::Tensor input) {
  auto sizes = input.sizes();
  int m = sizes[0];
  int n = sizes[1];

  auto output = torch::empty_like(input);
  dim3 block(32, 1);
  dim3 grid(m);

  auto *in_ptr = (const half4 *)input.data_ptr<at::Half>();
  auto *out_ptr = (half4 *)output.data_ptr<at::Half>();

  int cpt = n / 32;
  if (cpt == 4) {
    softmax_stored_locally_mutli_dim<4><<<grid, block>>>(in_ptr, out_ptr, m, n);
  } else if (cpt == 8) {
    softmax_stored_locally_mutli_dim<8><<<grid, block>>>(in_ptr, out_ptr, m, n);
  } else if (cpt == 16) {
    softmax_stored_locally_mutli_dim<16>
        <<<grid, block>>>(in_ptr, out_ptr, m, n);
  }

  return output;
}

torch::Tensor forward(torch::Tensor q, torch::Tensor k, torch::Tensor v) {}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &forward, "Flash attention forward");
  m.def("softmax", &softmax_forward, "Softmax forward");
}

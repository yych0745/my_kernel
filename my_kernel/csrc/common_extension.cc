#include <ATen/core/dispatch/Dispatcher.h>
#include <torch/all.h>
#include <torch/library.h>

#include "kernel.h"

TORCH_LIBRARY_FRAGMENT(my_kernel, m) {
    m.def("matmul(Tensor a, Tensor b, Tensor c) -> ()", &matmul);
    m.impl("matmul",torch::kCUDA, &matmul);
    m.def("matmul_warp(Tensor a, Tensor b, Tensor c) -> ()", &matmul_warp);
    m.impl("matmul_warp",torch::kCUDA, &matmul_warp);
    m.def("matmul_ptx(Tensor a, Tensor b, Tensor c) -> ()", &matmul_ptx);
    m.impl("matmul_ptx",torch::kCUDA, &matmul_ptx);
    m.def("matmul_naive(Tensor a, Tensor b, Tensor c) -> ()", &matmul_naive);
    m.impl("matmul_naive",torch::kCUDA, &matmul_naive);
}
REGISTER_EXTENSION(common_ops)
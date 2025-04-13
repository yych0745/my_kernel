#include <ATen/core/dispatch/Dispatcher.h>
#include <torch/all.h>
#include <torch/library.h>

#include "kernel.h"

TORCH_LIBRARY_FRAGMENT(my_kernel, m) {
    m.def("matmul(Tensor a, Tensor b, Tensor c) -> ()", &matmul);
    m.impl("matmul",torch::kCUDA, &matmul);
}
REGISTER_EXTENSION(common_ops)
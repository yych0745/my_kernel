#include <cuda_runtime.h>
#include <torch/all.h>
#include <mma.h>

constexpr int ceildiv(int a, int b) {
    return (a + b - 1) / b;
}
#include "utils.h"
// A: (rows, cols) 
// B: (cols, rows)
template <int block_n>  // 标准模板参数格式
__global__ void matmul_kernel(
    const half* a,
    const half* b,
    half* c,
    const int a_rows,
    const int a_cols,
    const int b_rows,
    const int b_cols
) {
    int block_id = blockIdx.x;
    int thread_id = threadIdx.x;
    int warp_id = thread_id / 32;
    // 一个block处理
    // block_n行 A 
    // 一列B, 列主序
    int N;
    if (block_n < a_rows) {
        N = a_cols * block_n;
    } else {
        N = a_rows * a_cols;
    }
    b = b + block_id * b_cols;
    int c_cols = b_rows;
    int c_rows = a_rows;
    int iterator = block_n;
    for (int i = thread_id; i < a_cols * a_rows; i += blockDim.x) {
        int a_row = i / a_cols;
        int a_col = i % a_cols;
        int b_row = block_id;
        int b_col = a_col;
        int c_row = a_row;
        int c_col = b_row;
        atomicAdd(&c[c_row * c_cols + c_col], a[i] * b[a_col]);
        // printf("i: %d, a[%d]: %f, b[%d]: %f, c[%d]: %f\n", i, i, a[i], a_col, b[a_col], c_row * c_cols + c_col, c[c_row * c_cols + c_col]);
    }
}

void matmul(
    torch::Tensor a,
    torch::Tensor b,
    torch::Tensor c) {
    int shared_mem_size = 48 * 1024;
    // std::cout << "shared_mem_size: " << shared_mem_size << " b.size(0): " << b.size(0) << "b.size(1): " << b.size(1) << std::endl;
    // cudaDeviceGetAttribute(&sms, shared_mem_size, dev)
    int blocks = b.size(1);
    // 单block处理4行
    matmul_kernel<8><<<blocks, 256, shared_mem_size, 0>>>(
            reinterpret_cast<half*>(a.data_ptr<at::Half>()),
            reinterpret_cast<half*>(b.data_ptr<at::Half>()),
            reinterpret_cast<half*>(c.data_ptr<at::Half>()),
            a.size(0),
            a.size(1),
            b.size(1),
            b.size(0));
        // 然后检查错误
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
        
        // 同步并再次检查错误
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        printf("CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}
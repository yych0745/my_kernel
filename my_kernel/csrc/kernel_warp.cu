#include "utils.h"
#define WMMA_M 16
#define WMMA_N 16
#define WMMA_K 16

#define WARP_SIZE 32

using namespace nvcuda;

// 256 bit = 32 bytes = 8个float
__global__ void matmul_kernel_warp(
    half *a,
    half *b,
    half *c,
    int M,
    int N,
    int K) {
        wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, half> c_frag;
        wmma::fill_fragment(c_frag, 0.0f);
        int K_Tiles = ceildiv(K, WMMA_K);
        int warp_row = blockIdx.x * WMMA_M;
        int warp_col = blockIdx.y * WMMA_N;
        if (warp_row >= M && warp_col >= N) {
            return;
        }
        wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> a_frag;
        wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::col_major> b_frag;
        for (int i = 0; i < K_Tiles; i++) {
            wmma::load_matrix_sync(a_frag, a + warp_row * K + i * WMMA_K, K);
            wmma::load_matrix_sync(b_frag, b + i * WMMA_K + warp_col * K, K);
            wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
        }
        wmma::store_matrix_sync(c + warp_row * N + warp_col, c_frag, N, wmma::mem_row_major);
}

void matmul_warp(
    torch::Tensor a,
    torch::Tensor b,
    torch::Tensor c) {
    assert(a.size(0) == b.size(1));
    assert(a.size(1) == b.size(0));

    dim3 block(WARP_SIZE);
    dim3 grid(ceildiv(a.size(0), WMMA_M), ceildiv(b.size(1), WMMA_N));
    // 单block处理4行
    matmul_kernel_warp<<<grid, block>>>(
        reinterpret_cast<half*>(a.data_ptr<at::Half>()),
        reinterpret_cast<half*>(b.data_ptr<at::Half>()),
        reinterpret_cast<half*>(c.data_ptr<at::Half>()),
            a.size(0),
            b.size(1),
            a.size(1)
        );
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
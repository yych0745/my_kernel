#include "utils.h"

using namespace nvcuda;
template <typename T, int n>
struct Vec {
  T elems[n];
  __device__ T& operator[](int i) {
    return elems[i];
  }
};

using FragA = Vec<half2, 4>;
using FragB = Vec<half2, 2>;
using FragC = Vec<float, 4>;
// m16n8k16 tensor core mma instruction with fp16 inputs and fp32 output/accumulation.
__device__ inline void mma(const FragA& a_frag, const FragB& frag_b, FragC& frag_c) {
    const uint32_t* a = reinterpret_cast<const uint32_t*>(&a_frag);
    const uint32_t* b = reinterpret_cast<const uint32_t*>(&frag_b);
    float* c = reinterpret_cast<float*>(&frag_c);
    asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
      "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
      : "=f"(c[0]), "=f"(c[1]), "=f"(c[2]), "=f"(c[3])
      :  "r"(a[0]),  "r"(a[1]),  "r"(a[2]),  "r"(a[3]),  "r"(b[0]),  "r"(b[1]),
         "f"(c[0]),  "f"(c[1]),  "f"(c[2]),  "f"(c[3])
    );
  }

  __device__ inline void cp_async4_stream(void* smem_ptr, const void* glob_ptr) {
    const int BYTES = 16;
    uint32_t smem = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
    asm volatile(
      "{\n"
      "   .reg .b64 p;\n"
      "   createpolicy.fractional.L2::evict_first.b64 p, 1.0;"
      "   cp.async.cg.shared.global.L2::cache_hint [%0], [%1], %2, p;\n"
      "}\n" :: "r"(smem), "l"(glob_ptr), "n"(BYTES)
    );
  }

  __device__ inline void cp_async4_pred(void* smem_ptr, const void* glob_ptr, bool pred = true) {
    const int BYTES = 16;
    uint32_t smem = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
    asm volatile(
      "{\n"
      "   .reg .pred p;\n"
      "   setp.ne.b32 p, %0, 0;\n"
      "   @p cp.async.cg.shared.global [%1], [%2], %3;\n"
      "}\n" :: "r"((int) pred), "r"(smem), "l"(glob_ptr), "n"(BYTES)
    );
  }
  
  // Instruction for loading a full 16x16 matrix fragment of operand A from shared memory, directly in tensor core layout.
__device__ inline void ldsm4(FragA& frag_a, const void* smem_ptr) {
    uint32_t* a = reinterpret_cast<uint32_t*>(&frag_a);
    uint32_t smem = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
    asm volatile(
      "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
      : "=r"(a[0]), "=r"(a[1]), "=r"(a[2]), "=r"(a[3]) : "r"(smem)
    );
  }
  // Instruction for loading a full 16x16 matrix fragment of operand A from shared memory, directly in tensor core layout.
  __device__ inline void ldsm2(FragB& frag_b, const void* smem_ptr) {
    uint32_t* a = reinterpret_cast<uint32_t*>(&frag_b);
    uint32_t smem = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
    asm volatile(
      "ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];\n"
      : "=r"(a[0]), "=r"(a[1]) : "r"(smem)
    );
  }



__global__ void matmul_kernel_ptx(
    half *a,
    half *b,
    half *c,
    int M,
    int N,
    int K) {
        int tid = threadIdx.x;
        extern __shared__ int4 sh[];
        int4* sh_a = sh;
        int4* sh_b = sh + (16 * 16);
        FragA a_frag;
        FragB b_frag[2];
        FragC c_frag[2];
        for(int i = 0; i < 4; i++) {
            c_frag[0][i] = 0.0f;
        }
        uint32_t RC[2] = {0, 0};
        if (tid == 0) {
            printf("a[0] = %f, a[1] = %f\n", __half2float(a[0]), __half2float(a[1]));
        }
        // global to shared
        cp_async4_pred(sh_a + tid, a + tid * 8);

        if (tid < 16) {
            cp_async4_stream(sh_b + tid, b + tid * 8);
        }
        // // load 16x16 matrix fragment of operand A from shared memory, directly in tensor core layout.
        asm volatile("cp.async.wait_all;");
        int transform_index[32] = {0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31};
        int transform_index_b[32] = {0, 2, 4, 6, 8, 10, 12, 14, 1, 3, 5, 7, 9, 11, 13, 15, 0, 2, 4, 6, 8, 10, 12, 14, 1, 3, 5, 7, 9, 11, 13, 15};
        ldsm4(a_frag, sh_a + transform_index[tid]);
        ldsm2(b_frag[0], sh_b + transform_index_b[tid]);


        if (tid == 0) {
            auto a_half = reinterpret_cast<half*>(sh_a);
            printf("a[0] = %f, a[1] = %f a[2] = %f, a[3] = %f\n", __half2float(a[0]), __half2float(a[1]), __half2float(a[2]), __half2float(a[3]));
            auto a_half2 = reinterpret_cast<half2*>(sh_a + transform_index[16]);
            printf("transform_index[17]: %d a_half2[0] = %f, %f\n", transform_index[16], __half2float(a_half[0]), __half2float(a_half[1]));
            printf("a_half2[1] = %f, %f\n", __half2float(a_half[2]), __half2float(a_half[3]));
            printf("a_half2[2] = %f, %f\n", __half2float(a_half[4]), __half2float(a_half[5]));
            printf("a_half2[3] = %f, %f\n", __half2float(a_half[6]), __half2float(a_half[7]));
            auto b_half = reinterpret_cast<half*>(sh_b + transform_index_b[1]);
            printf("b_half[0] = %f, %f\n", __half2float(b_half[0]), __half2float(b_half[1]));
            printf("b_half[1] = %f, %f\n", __half2float(b_half[2]), __half2float(b_half[3]));
            printf("b_half[2] = %f, %f\n", __half2float(b_half[4]), __half2float(b_half[5]));
            printf("b_half[3] = %f, %f\n", __half2float(b_half[6]), __half2float(b_half[7]));
            printf("b_half[4] = %f, %f\n", __half2float(b_half[8]), __half2float(b_half[9]));
            printf("b_half[5] = %f, %f\n", __half2float(b_half[10]), __half2float(b_half[11]));
            printf("b_half[6] = %f, %f\n", __half2float(b_half[12]), __half2float(b_half[13]));
            printf("b_half[7] = %f, %f\n", __half2float(b_half[14]), __half2float(b_half[15]));
            printf("b_half[8] = %f, %f\n", __half2float(b_half[16]), __half2float(b_half[17]));
            printf("b_half[9] = %f, %f\n", __half2float(b_half[18]), __half2float(b_half[19]));
            printf("b_half[10] = %f, %f\n", __half2float(b_half[20]), __half2float(b_half[21]));
            printf("b_half[11] = %f, %f\n", __half2float(b_half[22]), __half2float(b_half[23]));
            printf("b_half[12] = %f, %f\n", __half2float(b_half[24]), __half2float(b_half[25]));
            printf("b_half[13] = %f, %f\n", __half2float(b_half[26]), __half2float(b_half[27]));
            printf("b_half[14] = %f, %f\n", __half2float(b_half[28]), __half2float(b_half[29]));
            printf("b_half[15] = %f, %f\n", __half2float(b_half[30]), __half2float(b_half[31]));

        }

        // // mma
        mma(a_frag, b_frag[0], c_frag[0]);
        __syncthreads();
        printf("tid: %d a_frag[0] = %f, %f a_frag[1] = %f, %f a_frag[2] = %f, %f, a_frag[3] = %f, %f\n", tid, (__low2float(a_frag[0])), (__high2float(a_frag[0])), (__low2float(a_frag[1])), (__high2float(a_frag[1])), (__low2float(a_frag[2])), (__high2float(a_frag[2])), (__low2float(a_frag[3])), (__high2float(a_frag[3])));  
        printf("tid: %d b_frag[0] = %f, %f b_frag[1] = %f, %f\n", tid, __low2float(b_frag[0][0]), __high2float(b_frag[0][0]), __low2float(b_frag[0][1]), __high2float(b_frag[0][1]));
        half2 res11 = __halves2half2(__float2half(c_frag[0][0]), __float2half(c_frag[0][1]));
        half2 res12 = __halves2half2(__float2half(c_frag[0][2]), __float2half(c_frag[0][3]));
        printf("tid: %d res11: = %f, %f res12: = %f, %f\n", tid, __float2half(c_frag[0][0]), __float2half(c_frag[0][1]), __float2half(c_frag[0][2]), __float2half(c_frag[0][3]));
        int gl_stride_half2 = N / 2;
        ((half2*)(c))[tid / 4 * gl_stride_half2 + tid % 4] = res11;
        ((half2*)(c))[tid / 4 * gl_stride_half2 + tid % 4 + gl_stride_half2 * 8] = res12;
        

}



void matmul_ptx(
    torch::Tensor a,
    torch::Tensor b,
    torch::Tensor c) {
    assert(a.size(0) == b.size(1));
    assert(a.size(1) == b.size(0));

    dim3 block(32);
    dim3 grid(1);
    int shared_mem_size = 48 * 1024;
    // 单block处理4行
    matmul_kernel_ptx<<<grid, block, shared_mem_size>>>(
        reinterpret_cast<half*>(a.data_ptr<at::Half>()),
        reinterpret_cast<half*>(b.data_ptr<at::Half>()),
        reinterpret_cast<half*>(c.data_ptr()),
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
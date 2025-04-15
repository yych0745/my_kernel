import triton
import triton.testing
import torch
from my_kernel import kernel1
from my_kernel import kernel2


def ref_kernel(a, b):
    return torch.matmul(a, b.T)

"""
   batch_size       ref      kernel  kernel_warp
0        16.0  0.011712   76.834435     0.057280
1        32.0  0.009216   79.054588     0.056960
2        64.0  0.009536   96.781410     0.052288
3       128.0  0.010016  115.777565     0.053312
4       256.0  0.010672  177.770813     0.057632
5       512.0  0.012704  255.362686     0.070208
6      1024.0  0.014176  337.489899     0.096608
7      2048.0  0.019168  458.116669     0.187904
"""
@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["batch_size"],
        x_vals=[16, 32, 64, 128, 256, 512, 1024, 2048],
        x_log=False,
        line_arg="provider",
        line_vals=["ref", "kernel", "kernel_warp"],
        line_names=["ref", "kernel", "kernel_warp"],
        styles=[("blue", "-"), ("orange", "-"), ("green", "-")],
        ylabel="ms",
        plot_name="int8 scaled matmul",
        args={},
    )
)
def benchmark(provider, batch_size, N, K):
    M = batch_size
    a = torch.randn(M, K, dtype=torch.float16, device='cuda')
    b = torch.randn(N, K, dtype=torch.float16, device='cuda')
    c = torch.zeros(M, N, dtype=torch.float16, device='cuda')

    quantiles = [0.5, 0.2, 0.8]
    if provider == "ref":
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: ref_kernel(a, b),
            quantiles=quantiles,
        )
    if provider == "kernel":
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: kernel1(a, b, c),
            quantiles=quantiles,
        )
    if provider == "kernel_warp":
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: kernel2(a, b, c),
            quantiles=quantiles,
        )
    return ms, min_ms, max_ms

if __name__ == "__main__":
    for N in [16]:
        for K in [1024]:
            benchmark.run(
                print_data=True, show_plots=True, save_path="bench_res", N=N, K=K
            )
            # benchmark(N, K)
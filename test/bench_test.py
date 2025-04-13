import triton
import triton.testing
import torch
from my_kernel import kernel1


def ref_kernel(a, b):
    return torch.matmul(a, b.T)

def kernel(a, b, c):
    kernel1(a, b.T, c)

@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["batch_size"],
        x_vals=[1, 16, 32, 64, 128, 256, 512, 1024, 2048],
        x_log=False,
        line_arg="provider",
        line_vals=["ref", "kernel"],
        line_names=["ref", "kernel"],
        styles=[("blue", "-"), ("orange", "-")],
        ylabel="ms",
        plot_name="int8 scaled matmul",
        args={},
    )
)
def benchmark(provider, batch_size, N, K):
    M = batch_size
    a = torch.randn(M, K, dtype=torch.float, device='cuda')
    b = torch.randn(N, K, dtype=torch.float, device='cuda')
    c = torch.zeros(M, N, dtype=torch.float, device='cuda')
    scale_a = torch.randn((M,), device="cuda", dtype=torch.float32)
    scale_b = torch.randn((N,), device="cuda", dtype=torch.float32)
    bias = torch.randn((N,), device="cuda", dtype=torch.float16)

    quantiles = [0.5, 0.2, 0.8]
    if provider == "ref":
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: ref_kernel(a, b),
            quantiles=quantiles,
        )
    if provider == "kernel":
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: kernel(a, b, c),
            quantiles=quantiles,
        )
    return ms, min_ms, max_ms

if __name__ == "__main__":
    for N in [1]:
        for K in [1024]:
            benchmark.run(
                print_data=True, show_plots=True, save_path="bench_res", N=N, K=K
            )
            # benchmark(N, K)
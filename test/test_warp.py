# 矩阵乘法
import unittest
import torch
import itertools
from my_kernel import kernel2

class TestKernel(unittest.TestCase):
    DTYPES = [torch.float16]
    M = [16, 33]
    N = [128, 1024]
    K = [256, 4096]
    # M = [1]
    # N = [1024]
    # K = [512]
    SEEDS = [0]

    def ref_kernel(self, a, b):
        c = torch.matmul(a, b)
        return c

    def kernel(self, M, N, K, dtype, seed, device='cuda'):
        torch.manual_seed(seed)
        a = torch.randn(M, K, dtype=dtype, device='cuda')
        b = torch.randn(N, K, dtype=dtype, device='cuda')
        c = torch.zeros(M, N, dtype=dtype, device='cuda')
        kernel2(a, b.T, c)
        ref_c = self.ref_kernel(a, b.T)
        # self.assertTrue(torch.allclose(c_ref, ref_c, atol=1e-5))
        self.assertTrue(torch.mean(torch.abs(c.to(torch.float32) - ref_c.to(torch.float32)))
            / torch.mean(torch.abs(ref_c.to(torch.float32)))
            < 0.05)

    def test_kernel(self):
        for params in itertools.product(
            self.M,
            self.N,
            self.K,
            self.DTYPES,
            self.SEEDS,
        ):
            with self.subTest(
                M=params[0],
                N=params[1],
                K=params[2],
                dtype=params[3],
                seed=params[4],
            ):
                self.kernel(*params)
       

def test_one_kernel():
    torch.manual_seed(0)
    M = 16
    N = 1024 
    K = 32
    a = torch.randn(M, K, dtype=torch.half, device='cuda')
    b = torch.randn(N, K, dtype=torch.half, device='cuda')
    c = torch.zeros(a.shape[0], b.shape[0], dtype=torch.half, device='cuda')
    kernel2(a, b.T, c)
    ref_c = torch.matmul(a, b.T)
    print(c)
    print(ref_c)
    print(torch.mean(torch.abs(c.to(torch.float32) - ref_c.to(torch.float32)))
            / torch.mean(torch.abs(ref_c.to(torch.float32)))
            < 0.05)

if __name__ == "__main__":
    unittest.main()
    # test_one_kernel()

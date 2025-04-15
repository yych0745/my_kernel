import torch

def kernel1(a, b, c):
    return torch.ops.my_kernel.matmul.default(a, b, c)

def kernel2(a, b, c):
    return torch.ops.my_kernel.matmul_warp.default(a, b, c)

def kernel3(a, b, c):
    return torch.ops.my_kernel.matmul_ptx.default(a, b, c)

def kernel4(a, b, c):
    return torch.ops.my_kernel.matmul_naive.default(a, b, c)
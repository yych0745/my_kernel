import torch

def kernel1(a, b, c):
    return torch.ops.my_kernel.matmul.default(a, b, c)
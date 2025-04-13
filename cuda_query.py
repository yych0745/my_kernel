import torch

# 获取当前设备属性
prop = torch.cuda.get_device_properties(0)  # 0 是GPU设备ID
print(f"Shared Memory per Block: {prop.shared_mem_per_block / 1024} KB")
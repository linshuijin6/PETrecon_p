import torch
device = torch.device("cuda:7")
x = torch.zeros((92, 512, 512, 192), dtype=torch.float32, device=device)  # 调整分配大小
# 保留分配
del x  # 如果你需要释放这部分显存，可以删除此对象
torch.cuda.empty_cache()
1
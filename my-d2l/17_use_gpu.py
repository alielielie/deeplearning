# 计算设备
import torch
from torch import nn

print(torch.device('cpu'), torch.device('cuda'), torch.device('cuda:1'))

# 查询可用gpu的数量
print(torch.cuda.device_count())

# 这两个函数允许我们在不存在所需所有GPU的情况下运行代码
def try_gpu(i=0):
    """如果存在，则返回gpu(i)，否则返回cpu()"""
    if torch.cuda.device_count() >= i + 1:
        return torch.device(f'cuda:{i}')
    return torch.device('cpu')

def try_all_gpus():
    """返回所有可用的GPU，如果没有GPU，则返回[cpu(),]"""
    devices = [torch.device(f'cuda:{i}')
             for i in range(torch.cuda.device_count())]
    return devices if devices else [torch.device('cpu')]

print(try_gpu(), try_gpu(10), try_all_gpus())

# 查询张量所在的设备
x = torch.tensor([1, 2, 3])
print(x.device)

# 存储在GPU上
X = torch.ones(2, 3, device=try_gpu())
print(X)

# 神经网络与GPU
net = nn.Sequential(nn.Linear(3, 1))
net = net.to(device=try_gpu())

print(net(X))

# 确认模型参数存储在同一个GPU上
print(net[0].weight.data.device)
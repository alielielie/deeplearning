import torch
from torch import nn
from torch.nn import functional as F

# net = nn.Sequential(nn.Linear(20, 256), nn.ReLU(), nn.Linear(256, 10))

# X = torch.rand(2, 20)
# print(X)
# print(net(X))
#
# class MLP(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.hidden = nn.Linear(20, 256)
#         self.out = nn.Linear(256, 10)
#
#     def forward(self, X):
#         return self.out(F.relu(self.hidden(X)))
#
# net = MLP()
# print(net(X))

# 参数管理
# net = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 1))
# X = torch.rand(size=(2, 4))
# print(net(X))
# print(net[2].state_dict())
# print(type(net[2].bias))
# print(net[2].bias)
# print(net[2].bias.data)
# print(net[2].weight.grad is None)
# print(*[(name, param.shape) for name, param in net[0].named_parameters()])
# print(*[(name, param.shape) for name, param in net.named_parameters()])
# print(net.state_dict()['2.bias'].data)

# 从嵌套块收集参数
# def block1():
#     return nn.Sequential(nn.Linear(4, 8), nn.ReLU(),
#                          nn.Linear(8, 4), nn.ReLU())
#
# def block2():
#     net = nn.Sequential()
#     for i in range(4):
#         net.add_module(f'block {i}', block1())
#     return net
#
# rgnet = nn.Sequential(block2(), nn.Linear(4, 1))
# X = torch.rand(size=(2, 4))
# print(rgnet(X))
# print(rgnet)

# 内置初始化
net = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 1))

def init_normal(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, mean=0, std=0.01)
        nn.init.zeros_(m.bias)

net.apply(init_normal)
# print(net[0].weight.data[0], net[0].bias.data[0])

def init_constant(m):
    if type(m) == nn.Linear:
        nn.init.constant_(m.weight, 1)
        nn.init.zeros_(m.bias)
net.apply(init_constant)
# print(net[0].weight.data[0], net[0].bias.data[0])

# 对某些块应用不同的初始化方法
def init_xavier(m):
    if type(m) == nn.Linear:
        nn.init.xavier_uniform_(m.weight)
def init_42(m):
    if type(m) == nn.Linear:
        nn.init.constant_(m.weight, 42)

net[0].apply(init_xavier)
net[2].apply(init_42)
# print(net[0].weight.data[0])
# print(net[2].weight.data)

# 自定义初始化
def my_init(m):
    if type(m) == nn.Linear:
        print("Init", *[(name, param.shape)
                        for name, param in m.named_parameters()][0])
        nn.init.uniform_(m.weight, -10, 10)
        m.weight.data *= m.weight.data.abs() >= 5

# net.apply(my_init)
# print(net[0].weight[:2])
#
# net[0].weight.data[:] += 1
# net[0].weight.data[0, 0] = 42
# print(net[0].weight.data[0])

# 参数绑定
# shared = nn.Linear(8, 8)
# net = nn.Sequential(nn.Linear(4, 8), nn.ReLU(),
#                     shared, nn.ReLU(),
#                     shared, nn.ReLU(),
#                     nn.Linear(8, 1))
# X = torch.rand(size=(2, 4))
# net(X)
# print(net[2].weight.data[0] == net[4].weight.data[0])
# net[2].weight.data[0, 0] = 100
# print(net[2].weight.data[0] == net[4].weight.data[0])

# 自定义层
# 构造一个没有任何参数的自定义层
# class CenteredLayer(nn.Module):
#     def __init__(self):
#         super().__init__()
#
#     def forward(self, X):
#         return X - X.mean()
#
# layer = CenteredLayer()
# print(layer(torch.FloatTensor([1, 2, 3, 4, 5])))
#
# # 将层作为组件合并到更复杂的模型中
# net = nn.Sequential(nn.Linear(8, 128), CenteredLayer())
#
# Y = net(torch.rand(4, 8))
# print(Y.mean())
#
# # 带参数的层
# class MyLinear(nn.Module):
#     def __init__(self, in_units, units):
#         super().__init__()
#         self.weight = nn.Parameter(torch.randn(in_units, units))
#         self.bias = nn.Parameter(torch.randn(units,))
#     def forward(self, X):
#         linear = torch.matmul(X, self.weight.data) + self.bias.data
#         return F.relu(linear)
#
# linear = MyLinear(5, 3)
# print(linear.weight)
#
# # 使用自定义层直接执行前向传播计算
# print(linear(torch.rand(2, 5)))
#
# # 使用自定义层构建模型
# net = nn.Sequential(MyLinear(64, 8), MyLinear(8, 1))
# print(net(torch.rand(2, 64)))

# 读写文件
# 加载和保存张量
x = torch.arange(4)
torch.save(x, 'x-file')

x2 = torch.load('x-file')
print(x2)

# 存储一个张量列表，然后把它们读回内存
y = torch.zeros(4)
torch.save([x, y],'x-files')
x2, y2 = torch.load('x-files')
print((x2, y2))

# 写入或读取从字符串映射到张量的字典
mydict = {'x': x, 'y': y}
torch.save(mydict, 'mydict')
mydict2 = torch.load('mydict')
print(mydict2)

# 加载和保存模型参数
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden = nn.Linear(20, 256)
        self.output = nn.Linear(256, 10)

    def forward(self, x):
        return self.output(F.relu(self.hidden(x)))

net = MLP()
X = torch.randn(size=(2, 20))
Y = net(X)

# 将模型的参数存储在一个叫做“mlp.params”的文件中
torch.save(net.state_dict(), 'mlp.params')

# 实例化了原始多层感知机模型的一个备份。 直接读取文件中存储的参数
clone = MLP()
clone.load_state_dict(torch.load('mlp.params'))
clone.eval()

Y_clone = clone(X)
print(Y_clone == Y)
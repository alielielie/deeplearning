import torch

# # 标量
# x = torch.tensor([3.0])
# y = torch.tensor([2.0])
#
# # 向量
# m = torch.arange(4)
#
# # 矩阵
# A = torch.arange(20).reshape(5, 4)
# print(A)
# print(A.T)
#
# X = torch.arange(24).reshape(2, 3, 4)
# print(X)

# A = torch.arange(20, dtype=torch.float32).reshape(5, 4)
# B = A.clone()
# print(A)
# print(A + B)
# print(A * B)

# a = 2
# X = torch.arange(24).reshape(2, 3, 4)
# print(X)
# print(a + X)
# print(a * X)

# X = torch.arange(4, dtype=torch.float32)

a = torch.ones((2, 5, 4))
print(a.shape)
# print(a.sum(axis=[0, 2]).shape)
print(a.sum(axis=1))
print(a.sum(axis=1, keepdims=True).shape)
print(a.sum(axis=1, keepdims=True))

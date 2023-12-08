import torch
import os
import pandas as pd
#
# x = torch.arange(12)
# print(x)
# print(x.shape)
# print(x.numel())
# X = x.reshape(3, 4)
# print(X)
#
# print(torch.zeros((2, 3, 4)))
# print(torch.ones((2, 3, 4)))
#
# print(torch.tensor([[2, 1, 4, 3], [1, 2, 3, 4], [4, 3, 2, 1]]))
# print(torch.tensor([[[2, 1, 4, 3], [1, 2, 3, 4], [4, 3, 2, 1]]]).shape)
#
# x = torch.tensor([1.0, 2, 4, 8])
# y = torch.tensor([2, 2, 2, 2])
# print(x + y)
# print(x - y)
# print(x * y)
# print(x / y)
# print(x ** y)
#
# X = torch.arange(12, dtype=torch.float32).reshape((3, 4))
# Y = torch.tensor([[2.0, 1, 4, 3], [1, 2, 3, 4], [4, 3, 2, 1]])
# print(torch.cat((X, Y), dim=0))
# print(torch.cat((X, Y), dim=1))
# print(X == Y)
# print(X.sum())

# a = torch.arange(3).reshape((3, 1))
# b = torch.arange(2).reshape((1, 2))
# print(a)
# print(b)
# print(a + b)

# os.makedirs(os.path.join('.', 'data'), exist_ok=True)
data_file = os.path.join('.', 'data', 'house_tiny.csv')
# with open(data_file, 'w') as f:
#     f.write('NumRooms,Alley,Price\n')
#     f.write('NA,Pave,127500\n')
#     f.write('2,NA,106000\n')
#     f.write('4,NA,178100\n')
#     f.write('NA,NA,140000\n')
data = pd.read_csv(data_file)
print(data)
inputs, outputs = data.iloc[:, 0:2], data.iloc[:, 2]
inputs = inputs.fillna(inputs.mean())
inputs = pd.get_dummies(inputs, dummy_na=True)
print(inputs)
x, y = torch.tensor(inputs.values), torch.tensor(outputs.values)
print(x)
print(y)
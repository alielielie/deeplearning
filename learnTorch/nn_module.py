import torch
from torch import nn


class Zt(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        output = input + 1
        return output

zt = Zt()
x = torch.tensor(1.0)
output = zt(x)
print(output)
import torchvision
import torch
from torch import nn
from torch.nn import Linear
from torch.utils.data import DataLoader

dataset = torchvision.datasets.CIFAR10(root="./dataset", train=False,
                                       transform=torchvision.transforms.ToTensor(), download=True)
dataLoader = DataLoader(dataset=dataset, batch_size=64)

class Zt(nn.Module):
    def __init__(self):
        super(Zt, self).__init__()
        self.linear1 = Linear(196608, 10)

    def forward(self, input):
        output = self.linear1(input)
        return output

zt = Zt()

for data in dataLoader:
    imgs, targets = data
    print(imgs.shape)
    output = torch.flatten(imgs)
    print(output.shape)
    output = zt(output)
    print(output.shape)
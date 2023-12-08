import torch
import torchvision
from torch import nn
from torch.nn import ReLU, Sigmoid
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

input = torch.tensor([[1, -0.5],
                     [-1, 3]])

input = torch.reshape(input=input, shape=(-1, 1, 2, 2))
# print(input.shape)

dataset = torchvision.datasets.CIFAR10(root="./dataset", train=False,
                                       download=True, transform=torchvision.transforms.ToTensor())
dataLoader = DataLoader(dataset=dataset, batch_size=64)

class Zt(nn.Module):
    def __init__(self):
        super(Zt, self).__init__()
        self.relu1 = ReLU(inplace=False)
        self.sigmoid1 = Sigmoid()

    def forward(self, input):
        output = self.sigmoid1(input)
        return output

zt = Zt()

writer = SummaryWriter(log_dir="./logs_relu")
step = 0
for data in dataLoader:
    imgs, targets = data
    writer.add_images(tag="input", img_tensor=imgs, global_step=step)
    output = zt(imgs)
    writer.add_images(tag="output", img_tensor=output, global_step=step)
    step = step + 1

writer.close()


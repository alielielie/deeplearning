import torchvision
from torch import nn
from torch.nn import MaxPool2d
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

dataset = torchvision.datasets.CIFAR10(root="./dataset", train=False, download=True,
                                       transform=torchvision.transforms.ToTensor())
dataLoader = DataLoader(dataset=dataset, batch_size=64)

class Zt(nn.Module):
    def __init__(self):
        super(Zt, self).__init__()
        self.maxpool1 = MaxPool2d(kernel_size=3, ceil_mode=False)

    def forward(self, input):
        output = self.maxpool1(input)
        return output

zt = Zt()

writer = SummaryWriter(log_dir="./logs_maxpool")

step = 0
for data in dataLoader:
    imgs, targets = data
    writer.add_images(tag="input", img_tensor=imgs, global_step=step)
    output = zt(imgs)
    writer.add_images(tag="output", img_tensor=output, global_step=step)
    step = step + 1

writer.close()
import torch
from torch import nn
from torch.nn import Conv2d, MaxPool2d, Flatten, Linear, Sequential
from torch.utils.tensorboard import SummaryWriter


class Zt(nn.Module):
    def __init__(self):
        super(Zt, self).__init__()
        self.model1 = Sequential(
            Conv2d(in_channels=3, out_channels=32, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2)),
            MaxPool2d(kernel_size=2),
            Conv2d(in_channels=32, out_channels=32, kernel_size=(5, 5), padding=(2, 2)),
            MaxPool2d(kernel_size=2),
            Conv2d(in_channels=32, out_channels=64, kernel_size=(5, 5), padding=(2, 2)),
            MaxPool2d(kernel_size=2),
            Flatten(),
            Linear(in_features=1024, out_features=64),
            Linear(in_features=64, out_features=10)
        )

    def forward(self, x):
        x = self.model1(x)
        return x

zt = Zt()
print(zt)
input = torch.ones((64, 3, 32, 32))
output = zt(input)
print(output.shape)

writer = SummaryWriter(log_dir="logs_seq")
writer.add_graph(model=zt, input_to_model=input)
writer.close()

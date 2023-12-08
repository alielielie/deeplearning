import torchvision
from torch import nn
from torch.nn import Sequential, Conv2d, MaxPool2d, Flatten, Linear
from torch.utils.data import DataLoader

dataset = torchvision.datasets.CIFAR10(root="./dataset", train=False,
                                       transform=torchvision.transforms.ToTensor(), download=True)

data_loader = DataLoader(dataset=dataset, batch_size=1)

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
loss = nn.CrossEntropyLoss()
for data in data_loader:
    imgs, targets = data
    outputs = zt(imgs)
    result_loss = loss(outputs, targets)
    print("ok")
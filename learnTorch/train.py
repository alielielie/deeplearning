import torch
import torchvision
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from model import *

train_data = torchvision.datasets.CIFAR10(root="./dataset", train=True,
                                          transform=torchvision.transforms.ToTensor(), download=True)
test_data = torchvision.datasets.CIFAR10(root="./dataset", train=False,
                                         transform=torchvision.transforms.ToTensor(), download=True)

train_data_size = len(train_data)
test_data_size = len(test_data)

print("训练数据集的长度为：{}".format(train_data_size))
print("测试数据集的长度为：{}".format(test_data_size))

train_dataloader = DataLoader(dataset=train_data, batch_size=64)
test_dataloader = DataLoader(dataset=test_data, batch_size=64)

zt = Zt()

loss_fn = nn.CrossEntropyLoss()

# learning_rate = 0.01
learning_rate = 1e-2
optimizer = torch.optim.SGD(params=zt.parameters(), lr=learning_rate)

# 设置训练网络的一些参数
# 记录训练的次数
total_train_step = 0
# 记录测试的次数
total_test_step = 0
# 训练的轮数
epoch = 10

# 添加tensorboard
writer = SummaryWriter(log_dir="./logs_train")

for i in range(epoch):
    print("-----第{}轮训练开始-----".format(i + 1))
    # 训练步骤开始
    zt.train()
    for data in train_dataloader:
        imgs, targets = data
        outputs = zt(imgs)
        loss = loss_fn(outputs, targets)
        # 优化器优化模型
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # 记录训练步骤
        total_train_step = total_train_step + 1
        if total_train_step % 100 == 0:
            print("训练次数:{}, Loss:{}".format(total_train_step, loss.item()))
            writer.add_scalar(tag="train_loss", scalar_value=loss.item(), global_step=total_train_step)

    # 测试步骤开始
    zt.eval()
    total_test_loss = 0
    total_accuracy = 0
    with torch.no_grad():
        for data in test_dataloader:
            imgs, targets = data
            outputs = zt(imgs)
            loss = loss_fn(outputs, targets)
            total_test_loss = total_test_loss + loss.item()
            accuracy = (outputs.argmax(1) == targets).sum()
            total_accuracy = total_accuracy + accuracy
    print("整体测试集上的Loss:{}".format(total_test_loss))
    print("整体测试集上的正确率:{}".format(total_accuracy / test_data_size))
    writer.add_scalar(tag="test_loss", scalar_value=total_test_loss, global_step=total_test_step)
    writer.add_scalar(tag="test_accuracy", scalar_value=total_accuracy / test_data_size, global_step=total_test_step)
    total_test_step = total_test_step + 1

    torch.save(zt, "zt_{}.pth".format(i))
    print("模型已保存")

writer.close()




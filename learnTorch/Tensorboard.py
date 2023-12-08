from torch.utils.tensorboard import SummaryWriter
import numpy as np
from PIL import Image

writer = SummaryWriter("logs")

image_path = "dataset/val/ants/800px-Meat_eater_ant_qeen_excavating_hole.jpg"
imgPIL = Image.open(image_path)
img_array = np.array(imgPIL)
# print(img_array.shape)

writer.add_image("train", img_array, 1, dataformats='HWC')

for i in range(100):
    writer.add_scalar("y=2x", 3*i, i)


writer.close()
import os.path
import matplotlib.pyplot as plt
import json
import numpy as np
from fn_generate_dataset import AutocorrDataset, compute_autocorr, normalization
from fn_loss import *
from fn_train import fn_train_Pix2Pix, fn_train_CycleGAN
import itertools
from model.Pix2Pix import Generator, Discriminator
from model.CycleGAN import Generator
from scipy.ndimage import zoom
from PIL import Image
def test():
    # 创建一个大小为 128x128 的全零张量
    tensor = torch.zeros(128, 128)
    # 计算中心区域的开始索引和结束索引
    start_index = (128 - 64) // 2  # 这将是 32
    end_index = start_index + 64  # 这将是 96
    # 将中心的 64x64 区域设置为 1
    tensor[start_index:end_index, start_index:end_index] = 1
    tensor = tensor.unsqueeze(0)
    tensor = tensor.unsqueeze(0)
    # 读取图片
    image = Image.open('../assets/number4_256.jpg')
    # 将图片转换为灰度图像
    gray_image = image.convert('L')
    # 将灰度图像转换为NumPy矩阵
    image_array = np.array(gray_image)
    image_array = zoom(image_array, zoom=0.5)
    x = compute_autocorr(image_array)
    x=torch.from_numpy(x).float()
    x=x.unsqueeze(0)
    x = x.unsqueeze(0)
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Using {device} device")
    x=x.to(device)
    tensor = tensor.to(device)
    num_epochs=10000
    model = Generator().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0002)
    criterion = torch.nn.MSELoss()
    running_loss=0.0
    for epoch in range(num_epochs):
        outputs = model(x)*tensor
        outputs = torch.clamp(outputs, min=0)
        outputs_ac = compute_autocorr(outputs)
        loss= criterion(outputs_ac, x)
        running_loss += loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if (epoch + 1) % (num_epochs / 10) == 0:
            print('epoch_loss: %.4f', loss.item())
            running_loss = 0.0
    outputs_show = outputs.squeeze()
    plt.imshow(outputs_show.cpu().detach().numpy(), cmap='gray')
    plt.colorbar()  # 可选，为图像添加颜色条
    plt.savefig('../assets/number4_untrained_phase_retrieval.png', transparent= True)
    plt.show()
if __name__ == "__main__":
    test()
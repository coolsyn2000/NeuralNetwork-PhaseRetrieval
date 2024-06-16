import os

import PIL.Image
import matplotlib.pyplot as plt

import numpy as np
import torch.fft

from fn_loss import *

from model.LightUnet import Unet

from scipy.ndimage import zoom
from PIL import Image
from scipy.ndimage import gaussian_filter
from fn_generate_dataset import compute_autocorr


def load_data(path, crop_size=1024, filter=None):
    img = PIL.Image.open(path)
    image_array = np.array(img)
    image_array = image_array.astype(np.float32)

    #crop_size = 1024
    # 计算中心区域的起始和结束索引
    start_row = (image_array.shape[0] - crop_size) // 2
    end_row = start_row + crop_size
    start_col = (image_array.shape[1] - crop_size) // 2
    end_col = start_col + crop_size

    # 截取中心的区域
    image_array = image_array[start_row:end_row, start_col:end_col]

    # min-max normalization

    min_value = image_array.min()
    max_value = image_array.max()

    # 将矩阵的值缩放到 [0, 1] 之间
    image_array = (image_array - min_value) / (max_value - min_value)

    if filter is not False:
        image_array = gaussian_filter(image_array, sigma=1)

    image_array = zoom(image_array, zoom=0.5)
    image_torch = torch.from_numpy(image_array)
    image_torch = image_torch.unsqueeze(0)
    image_torch = image_torch.unsqueeze(0)
    return image_torch


def run_phase_retrieval(data_path, model_input):
    # fix random seed
    seed = 999
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True

    data_torch = load_data(data_path, crop_size=1024, filter=True)

    dim = data_torch.size(2)
    # 创建suppmat
    # 创建一个大小为 128x128 的全零张量
    tensor = torch.zeros(1, 1, dim, dim)

    supp_size = 100
    # 计算中心区域的开始索引和结束索引
    start_index = (dim - supp_size) // 2  # 这将是 32
    end_index = start_index + supp_size  # 这将是 96

    # 将中心的 64x64 区域设置为 1
    tensor[:, :, start_index:end_index, start_index:end_index] = 1

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Using {device} device")

    tensor = tensor.to(device)
    model = Unet().to(device)

    inputs = data_torch.to(device)

    if model_input == "speckle":
        inputs_ft = torch.abs(torch.fft.fft2(data_torch))
        inputs_ac = torch.abs(torch.fft.ifft2(torch.abs(torch.fft.fftn(data_torch)) ** 2))
        inputs_ac = inputs_ac - torch.min(inputs_ac)
        inputs_ac = compute_autocorr(data_torch)

    if model_input == "autocorr":
        inputs_ft = torch.sqrt(torch.abs(torch.fft.fft2(torch.fft.ifftshift(data_torch))))
        inputs_ac = data_torch
    inputs_ft = inputs_ft.to(device)
    inputs_ac = inputs_ac.to(device)

    num_epochs = 500

    optimizer = torch.optim.Adam(model.parameters(), lr=0.02)

    criterion = torch.nn.L1Loss()
    running_loss = 0.0

    for epoch in range(num_epochs):

        outputs = model(inputs) * tensor

        outputs_ft = torch.abs(torch.fft.fft2(outputs))
        loss = criterion(outputs_ft, inputs_ft)

        #outputs_ac = torch.fft.ifftshift(torch.abs(torch.fft.ifftn(torch.abs(torch.fft.fftn(outputs)) ** 2)))
        #outputs_ac = outputs_ac-torch.min(outputs_ac)
        #loss = criterion(outputs_ac, inputs_ac)

        running_loss += loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (epoch + 1) % (num_epochs / 10) == 0:
            print('epoch_loss: %.4f ' % loss.item())

            running_loss = 0.0

    outputs_show = outputs.squeeze()
    outputs_show = outputs_show.cpu().detach().numpy()
    plt.imshow(outputs_show, cmap='gray')
    plt.colorbar()
    plt.tight_layout()
    plt.savefig('../assets/single_shot_untrained_phase_retrieval.png', transparent=True)
    plt.show()


if __name__ == "__main__":
    run_phase_retrieval('../assets/2_speckle.bmp',model_input='speckle')

    #run_phase_retrieval('./TEST.png', model_input='autocorr')

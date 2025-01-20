import os
import matplotlib.pyplot as plt

import numpy as np
import torch

from fn_loss import *

from model.SIREN import Unet, MLPToCNN

from scipy.ndimage import zoom
from PIL import Image
from scipy.ndimage import gaussian_filter


def read_bmp_images_from_folder(folder_path):
    # 初始化一个空列表来保存所有的NumPy矩阵
    image_array_list = []
    image_torch_list = []

    # 遍历文件夹中的所有文件
    for filename in os.listdir(folder_path):
        if filename.endswith("speckle.bmp"):  # 只处理 .bmp 文件
            # 构建完整的文件路径
            file_path = os.path.join(folder_path, filename)

            # 使用Pillow读取图片
            with Image.open(file_path) as img:
                # 将图片转换为NumPy数组
                image_array = np.array(img)
                image_array = image_array.astype(np.float32)

                crop_size = 1024
                # 计算中心区域的起始和结束索引
                start_row = (image_array.shape[0] - crop_size) // 2
                end_row = start_row + crop_size
                start_col = (image_array.shape[1] - crop_size) // 2
                end_col = start_col + crop_size

                # 截取中心的区域
                image_array = image_array[start_row:end_row, start_col:end_col]

                min_value = image_array.min()
                max_value = image_array.max()

                # 将矩阵的值缩放到 [0, 1] 之间
                image_array = (image_array - min_value) / (max_value - min_value)

                image_array = gaussian_filter(image_array, sigma=1)

                # plt.imshow(image_array)
                # plt.show()
                image_array = zoom(image_array, zoom=0.25)
                image_torch = torch.from_numpy(image_array)
                image_torch = image_torch.unsqueeze(0)
                image_torch = image_torch.unsqueeze(0)

                # 将NumPy数组添加到列表中
                image_array_list.append(image_array)
                image_torch_list.append(image_torch)

    return image_array_list, image_torch_list


def test():
    seed=666
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True

    # 创建一个大小为 128x128 的全零张量
    tensor = torch.zeros(256, 256)

    supp_size_w = 20
    supp_size_h = 30
    # 计算中心区域的开始索引和结束索引
    start_index_w = (256 - supp_size_w) // 2  # 这将是 32
    end_index_w = start_index_w + supp_size_w  # 这将是 96

    start_index_h = (256 - supp_size_h) // 2  # 这将是 32
    end_index_h = start_index_h + supp_size_h  # 这将是 96

    # 将中心的 64x64 区域设置为 1
    tensor[start_index_h:end_index_h, start_index_w:end_index_w] = 1
    tensor = tensor.unsqueeze(0)
    tensor = tensor.unsqueeze(0)

    # 读取图片
    folder_path = '../assets/'  # 替换为实际的文件夹路径
    speckle_array_list, speckle_torch_list, = read_bmp_images_from_folder(folder_path)

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Using {device} device")
    #
    # x=x.to(device)

    speckle_torch_list = [tensor.to(device) for tensor in speckle_torch_list]

    speckle_ft = [torch.abs(torch.fft.fft2(tensor)) for tensor in speckle_torch_list]
    speckle_pha = [torch.angle(torch.fft.fft2(tensor)) for tensor in speckle_torch_list]
    tensor = tensor.to(device)
    num_epochs = 3000
    model = MLPToCNN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    criterion = torch.nn.MSELoss()
    running_loss = 0.0
    losses = []
    for epoch in range(num_epochs):
        #for i in range(len(speckle_torch_list)):
        # outputs = model(all_one_tensor_gpu)
        outputs = model(torch.tensor([[1.0]]).to(device))
        loss=0.0
        for i in range(1):
            obj_pha = speckle_pha[i] - outputs
            obj = torch.real(torch.fft.ifft2((speckle_ft[i] * torch.exp(1j * obj_pha))))
            obj = obj * tensor
            obj[obj < 0] = 0
            outputs_ft = torch.abs(torch.fft.fft2(obj))
            # speckles_ft = torch.abs(torch.fft.fft2(speckle_torch_list[i]))
            loss += criterion(outputs_ft, speckle_ft[i])
        running_loss += loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        if (epoch + 1) % (num_epochs / 10) == 0:
            print('epoch_loss: ', loss.item())

            running_loss = 0.0

    fig, axes = plt.subplots(1, 5, figsize=(20, 4))

    model.eval()
    matrix_list = []

    supp_size_w = 120
    supp_size_h = 120
    # 计算中心区域的开始索引和结束索引
    start_index_w = (256 - supp_size_w) // 2  # 这将是 32
    end_index_w = start_index_w + supp_size_w  # 这将是 96

    start_index_h = (256 - supp_size_h) // 2  # 这将是 32
    end_index_h = start_index_h + supp_size_h  # 这将是 96

    for i in range(len(speckle_torch_list)):
        outputs = model(torch.tensor([[1.0]]).to(device))
        obj_pha = speckle_pha[i] - outputs
        obj_pha[obj_pha<-torch.pi] = obj_pha[obj_pha < -torch.pi]+2*torch.pi
        obj_pha[obj_pha > torch.pi] = obj_pha[obj_pha > torch.pi] - 2 * torch.pi
        obj = torch.real(torch.fft.ifft2((speckle_ft[i] * torch.exp(1j * obj_pha))))
        #obj = obj * tensor
        obj[obj < 0] = 0

        outputs_show = obj.squeeze()
        outputs_show = outputs_show.cpu().detach().numpy()
        matrix_list.append(outputs_show)

    for ax, matrix in zip(axes, matrix_list):
        img= ax.imshow(matrix[start_index_h:end_index_h, start_index_w:end_index_w], cmap='gray')  # 使用灰度颜色映射
        ax.axis('off')  # 隐藏坐标轴
        cbar = plt.colorbar(img, ax=ax, orientation='vertical')

    plt.tight_layout()
    plt.title('Reconstructed Images')
    plt.savefig('./OTF_DIP_Reconstructed_Images.png', transparent=True)
    plt.show()

    plt.imshow(torch.fft.ifftshift(outputs).squeeze().cpu().detach().numpy())
    plt.colorbar()
    plt.tight_layout()
    plt.title('PhaseMap of OTF')
    plt.savefig('./OTF_DIP_PhaseMap.png', transparent=True)
    plt.show()
    #plt.figure(figsize=(10, 5))
    plt.tight_layout()
    plt.plot(losses, label='Loss', color='blue')
    plt.title('Training Loss Over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('./OTF_DIP_Loss.png', transparent=True)
    plt.show()

if __name__ == "__main__":
    test()

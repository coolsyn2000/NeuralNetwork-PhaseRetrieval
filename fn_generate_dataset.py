import os

import numpy as np
import torch
from torch.utils.data import Dataset, random_split, DataLoader
from torchvision import datasets, transforms


# Compute the autocorr of MNIST image
# def compute_autocorr(image):
#     ft = np.fft.fft2(image)
#     autocorr = np.fft.ifft2(np.abs(ft) ** 2)
#     autocorr = abs(np.fft.fftshift(autocorr))
#     return autocorr


def compute_autocorr(image):
    # 检查输入类型，以决定使用的是NumPy还是PyTorch函数
    if isinstance(image, np.ndarray):
        # 使用NumPy进行计算
        ft = np.fft.fft2(image)
        autocorr = np.fft.ifft2(np.abs(ft) ** 2)
        autocorr = np.abs(np.fft.fftshift(autocorr))
    elif isinstance(image, torch.Tensor):
        # 确保张量在GPU上
        image = image.to(device='cuda')

        # 使用PyTorch进行计算
        ft = torch.fft.fftn(image)
        autocorr = torch.fft.ifftn(torch.abs(ft) ** 2)
        autocorr = torch.abs(torch.fft.fftshift(autocorr))
    else:
        raise TypeError("Unsupported input type. Expected numpy.ndarray or torch.Tensor")

    return autocorr

# [0,1] minmax
def normalization(data):
    _range = np.max(data) - np.min(data)
    return (data - np.min(data)) / _range


# Process and load MNIST dataset
class MNISTDataset(Dataset):
    def __init__(self, dataset):

        transform_image = transforms.Compose([
            # resize image to [128,128]
            transforms.Resize((128, 128)),

            transforms.ToTensor(),
        ])

        transform_autocorr = transforms.Compose([
            # pad image to [56,56]
            transforms.Pad(padding=14, fill=0, padding_mode='constant'),
            # resize image to [128,128]
            transforms.Resize((128, 128)),

            transforms.ToTensor(),
        ])

        self.dataset = dataset

        self.transform_image = transform_image
        self.transform_autocorr = transform_autocorr

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        # 获取图像及其标签（在这个案例中，标签未使用）
        img, _ = self.dataset[idx]
        img_pad = self.transform_autocorr(img)
        img = self.transform_image(img)
        # 计算自相关并归一化
        autocorr = normalization(compute_autocorr(img_pad.numpy())).squeeze()

        autocorr = autocorr[np.newaxis, :, :]

        autocorr = torch.tensor(autocorr, dtype=torch.float32)

        return autocorr, img


# Load custom MNIST_autocorr dataset
class AutocorrDataset(Dataset):
    def __init__(self, dataset_dict):
        self.images = dataset_dict['image']
        self.autocorrs = dataset_dict['autocorr']

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        autocorr = self.autocorrs[idx]

        return autocorr, image


# Generate autocorr dataset function
def generate_autocorr_dataset(custom_root_path,name_dataset="MNIST"):
    os.makedirs(custom_root_path, exist_ok=True)

    if name_dataset == "MNIST":
        test_dataset = datasets.MNIST(root=custom_root_path, train=False, download=True)
    if name_dataset == "FashionMNIST":
        test_dataset = datasets.FashionMNIST(root=custom_root_path, train=False, download=True)

    # load and process MNIST_test dataset
    MNIST_autocorr_dataset = MNISTDataset(test_dataset)

    # split to train and test dataset
    train_ac_dataset, test_ac_dataset = random_split(MNIST_autocorr_dataset, [9000, 1000])

    return train_ac_dataset, test_ac_dataset


# generate and save MNIST_autocorr dataset
def gen_save_dataset(datasets_path='./datasets'):

    # dataset have been created or not
    if os.path.exists(datasets_path) and os.path.exists('./datasets/MNIST_autocorr_test.pt'):
        print(f"The path {datasets_path} exists. Skip download dataset")
    else:
        print(f"The path {datasets_path} does not exist. Begin to create datasets")

        # generate autocorr dataset
        train_dataset, test_dataset = generate_autocorr_dataset(datasets_path)
        train_images = []
        train_autocorrs = []

        for autocorr, img in train_dataset:
            train_images.append(img)
            train_autocorrs.append(autocorr)

        dataset_save_path = './datasets/MNIST_autocorr_train.pt'
        torch.save({'autocorr': train_autocorrs, 'image': train_images}, dataset_save_path)

        test_images = []
        test_autocorrs = []

        for autocorr, img in test_dataset:
            test_images.append(img)
            test_autocorrs.append(autocorr)

        dataset_save_path = './datasets/MNIST_autocorr_test.pt'
        torch.save({'autocorr': test_autocorrs, 'image': test_images}, dataset_save_path)

        print(f"Dataset saved to {dataset_save_path}")

# load MNIST_autocorr dataset
def load_dataset(train_dataset_save_path='./datasets/MNIST_autocorr_train.pt',
                 test_dataset_save_path='./datasets/MNIST_autocorr_test.pt',batch_size = 64):
    loaded_train_data = torch.load(train_dataset_save_path)
    loaded_test_data = torch.load(test_dataset_save_path)

    train_dataset = AutocorrDataset(loaded_train_data)
    test_dataset = AutocorrDataset(loaded_test_data)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_dataloader, test_dataloader


if __name__ == "__main__":
    gen_save_dataset(datasets_path='./datasets')

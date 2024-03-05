import torch
import torch.fft as fft
import torch.nn.functional as F
from pytorch_msssim import ssim

def npcc(input, target):
    # 计算输入图像和目标图像的均值
    input_mean = torch.mean(input)
    target_mean = torch.mean(target)

    # 计算输入图像和目标图像的标准差
    input_std = torch.std(input)
    target_std = torch.std(target)

    # 计算协方差
    covariance = torch.mean((input - input_mean) * (target - target_mean))

    # 计算皮尔逊相关系数
    pearson_corr = covariance / (input_std * target_std)

    return -pearson_corr


def psnr(image1, image2):
    mse = F.mse_loss(image1, image2)  # 计算均方误差（Mean Squared Error）
    psnr = 20 * torch.log10(1.0 / torch.sqrt(mse))  # 计算 PSNR 分数
    return psnr


def image_autocorrelation(image):
    # 将图像转换为复数类型，进行傅里叶变换
    f_image = fft.fftn(image, dim=(-2, -1))

    # 计算图像的功率谱密度
    power_spectrum = torch.abs(f_image) ** 2

    # 进行反向傅里叶变换得到自相关函数
    autocorrelation = fft.ifftn(power_spectrum, dim=(-2, -1))
    autocorrelation = torch.abs(autocorrelation)
    # autocorrelation = (autocorrelation - autocorrelation.min()) / (autocorrelation.max() - autocorrelation.min())

    return autocorrelation


def autocorrelation_mae_loss(input, target):
    # Ensure the input and target have the same dimensions
    if input.size() != target.size():
        raise ValueError("Input and target must have the same dimensions.")

    # Calculate the autocorrelation of input and target images
    input_autocorr = image_autocorrelation(input)
    target_autocorr = image_autocorrelation(target)

    # Calculate the MAE loss between the autocorrelation functions
    loss = F.l1_loss(input_autocorr, target_autocorr)

    return loss

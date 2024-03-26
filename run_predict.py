import torch
import numpy as np
import fn_generate_dataset
import matplotlib.pyplot as plt
import torchvision
from fn_loss import npcc,ssim,psnr
from model.Pix2Pix import Generator, Discriminator


def test_from_dataset():
    fn_generate_dataset.gen_save_dataset()
    _, test_dataloader = fn_generate_dataset.load_dataset()

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Using {device} device")

    net = Generator()
    net.load_state_dict(torch.load('./assets/GAN-PhaseRetrieval-MNIST_pretrained.pth'))

    with torch.no_grad():
        dataiter = iter(test_dataloader)
        autocorr, image = next(dataiter)
        image_fake = net(autocorr)

        SSIM_value = ssim(image_fake, image, data_range=1.0, size_average=True)
        MAE_value = torch.nn.L1Loss()(image_fake, image)
        PSNR_value = psnr(image_fake, image)

    plt.figure(figsize=(10,10))

    plt.subplot(3,1,1)
    plt.title('autocorr')
    plt.imshow(np.transpose(torchvision.utils.make_grid(autocorr[:4]), (1, 2, 0)))

    plt.subplot(3,1,2)
    plt.title('image_label')
    plt.imshow(np.transpose(torchvision.utils.make_grid(image[:4]), (1, 2, 0)))

    plt.subplot(3,1,3)
    plt.title('image_fake')
    plt.imshow(np.transpose(torchvision.utils.make_grid(image_fake[:4]), (1, 2, 0)))
    plt.savefig('./assets/model_prediction.png', transparent=True)
    plt.show()

def test_from_sample():

    net = Generator()
    net.load_state_dict(torch.load('./assets/GAN-PhaseRetrieval-MNIST_pretrained.pth'))

    sample = torch.load('test_autocorr_4.pt')

    with torch.no_grad():
        image_fake = net(sample)

    plt.imshow(image_fake.squeeze().detach(), cmap='gray')
    plt.colorbar()
    plt.savefig('./assets/model_prediction_4.png', transparent=True)
    plt.show()

if __name__=="__main__":
    test_from_sample()



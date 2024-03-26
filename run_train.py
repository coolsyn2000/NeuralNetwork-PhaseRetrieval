import os.path
import matplotlib.pyplot as plt
import json
import numpy as np
import fn_generate_dataset
from fn_loss import *
from fn_train import fn_train_Pix2Pix, fn_train_CycleGAN
import itertools
from model.Pix2Pix import Generator, Discriminator
from model.CycleGAN import Generator


def train_Pix2Pix():
    fn_generate_dataset.gen_save_dataset()
    train_dataloader, test_dataloader = fn_generate_dataset.load_dataset()

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Using {device} device")

    generator = Generator().to(device)
    discriminator = Discriminator().to(device)

    opt_disc = torch.optim.Adam(discriminator.parameters(), lr=0.0001, betas=(0.5, 0.999), )
    opt_gen = torch.optim.Adam(generator.parameters(), lr=0.001, betas=(0.5, 0.999))

    g_scaler = torch.cuda.amp.GradScaler()
    d_scaler = torch.cuda.amp.GradScaler()

    num_epochs = 50

    loss_history = {
        'Train_GLoss': np.ones(num_epochs),
        'Train_DLoss': np.ones(num_epochs),
        'Test_SSIM': np.ones(num_epochs),
        'Test_MAE': np.ones(num_epochs),
        'Test_PSNR': np.ones(num_epochs)
    }

    if not os.path.exists('./results/'):
        os.mkdir('./results/')

    for epoch in range(num_epochs):
        loss_history = fn_train_Pix2Pix(train_dataloader, test_dataloader, device, epoch, discriminator, generator,
                                        opt_disc, opt_gen,
                                        g_scaler, d_scaler, loss_history)

    torch.save(generator.state_dict(), f'./results/GAN_PhaseRetrieval_MNIST_final.pth')

    try:
        loss_history['Train_GLoss'] = loss_history['Train_GLoss'].tolist()
        loss_history['Train_DLoss'] = loss_history['Train_DLoss'].tolist()
        loss_history['Test_SSIM'] = loss_history['Test_SSIM'].tolist()
        loss_history['Test_MAE'] = loss_history['Test_MAE'].tolist()
        loss_history['Test_PSNR'] = loss_history['Test_PSNR'].tolist()

        with open('./results/GAN_PhaseRetrieval_MNIST_history.json', 'w') as fp:
            json.dump(loss_history, fp)
    except:
        pass

    if not os.path.exists('./results/figs/'):
        os.mkdir('./results/figs/')

    plt.plot(loss_history['Test_PSNR'])
    plt.title('Model PSNR')
    plt.savefig('./results/figs/PSNR_curve.png', transparent=True)
    plt.show()

    plt.plot(loss_history['Test_SSIM'])
    plt.title('Model SSIM')
    plt.savefig('./results/figs/SSIM_curve.png', transparent=True)
    plt.show()

    plt.plot(loss_history['Test_MAE'])
    plt.title('Model MAE')
    plt.savefig('./results/figs/MAE_curve.png', transparent=True)
    plt.show()


def train_CycleGAN():
    fn_generate_dataset.gen_save_dataset()
    train_dataloader, test_dataloader = fn_generate_dataset.load_dataset(batch_size=8)

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Using {device} device")

    netG_A2B = Generator().to(device)
    netG_B2A = Generator().to(device)
    # netD_A = Discriminator().to(device)
    # netD_B = Discriminator().to(device)

    netD_A = Generator().to(device)
    netD_B = Generator().to(device)

    netG_A2B_scaler = torch.cuda.amp.GradScaler()
    netG_B2A_scaler = torch.cuda.amp.GradScaler()

    netD_B_scaler = torch.cuda.amp.GradScaler()
    netD_A_scaler = torch.cuda.amp.GradScaler()


    optimizer_G = torch.optim.Adam(itertools.chain(netG_A2B.parameters(), netG_B2A.parameters()), lr=0.0002,
                                   betas=(0.5, 0.999))
    optimizer_D_A = torch.optim.Adam(netD_A.parameters(), lr=0.0002, betas=(0.5, 0.999))
    optimizer_D_B = torch.optim.Adam(netD_B.parameters(), lr=0.0002, betas=(0.5, 0.999))

    num_epochs = 50

    loss_history = {
        'Train_GLoss': np.ones(num_epochs),
        'Train_DLoss': np.ones(num_epochs),
        'Test_SSIM': np.ones(num_epochs),
        'Test_MAE': np.ones(num_epochs),
        'Test_PSNR': np.ones(num_epochs)
    }

    if not os.path.exists('./results/'):
        os.mkdir('./results/')

    for epoch in range(num_epochs):
        loss_history = fn_train_CycleGAN(train_dataloader, test_dataloader, device, epoch, netG_A2B, netG_B2A, netD_A, netD_B
                                        , optimizer_G, optimizer_D_A, optimizer_D_B,netG_A2B_scaler,netG_B2A_scaler,netD_A_scaler,netD_B_scaler)

if __name__ == "__main__":
    train_Pix2Pix()
    #train_CycleGAN()









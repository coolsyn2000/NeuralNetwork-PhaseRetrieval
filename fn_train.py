import os.path

import torch
import torchvision
from tqdm import tqdm
from fn_loss import psnr, npcc, ssim


def train(train_dataloader, test_dataloader, device, epoch, disc, gen, opt_disc, opt_gen, g_scaler, d_scaler,
          loss_history):

    Train_GLoss, Train_DLoss, SSIM_metrics, MAE_metrics, PSNR_metrics = 0.0, 0.0, 0.0, 0.0, 0.0

    loop = tqdm(train_dataloader, leave=True)

    for idx, (x, y) in enumerate(loop):
        x = x.to(device)
        y = y.to(device)

        # Train Discriminator
        with torch.cuda.amp.autocast():
            y_fake = gen(x)
            D_real = disc(x, y)
            D_real_loss = torch.nn.BCEWithLogitsLoss()(D_real, torch.ones_like(D_real)) * 1
            D_fake = disc(x, y_fake.detach())
            D_fake_loss = torch.nn.BCEWithLogitsLoss()(D_fake, torch.zeros_like(D_fake)) * 1
            D_loss = (D_real_loss + D_fake_loss)

        Train_DLoss += D_loss.item()
        disc.zero_grad()
        d_scaler.scale(D_loss).backward()
        d_scaler.step(opt_disc)
        d_scaler.update()

        # Train generator
        with torch.cuda.amp.autocast():
            D_fake = disc(x, y_fake)
            G_fake_loss = torch.nn.BCEWithLogitsLoss()(D_fake, torch.ones_like(D_fake))
            L2 = torch.nn.MSELoss()(y_fake, y) * 100
            G_loss = G_fake_loss + L2

        Train_GLoss += G_loss.item()
        opt_gen.zero_grad()
        g_scaler.scale(G_loss).backward()
        g_scaler.step(opt_gen)
        g_scaler.update()

        if idx % 10 == 0:
            loop.set_postfix(
                D_real=torch.sigmoid(D_real).mean().item(),
                D_fake=torch.sigmoid(D_fake).mean().item(),
            )

    Train_DLoss /= len(train_dataloader)
    Train_GLoss /= len(train_dataloader)
    loss_history['Train_DLoss'][epoch] = Train_DLoss
    loss_history['Train_GLoss'][epoch] = Train_GLoss
    print(f'Epoch [{epoch + 1}], Train_DLoss: {Train_DLoss:.4f}, Train_GLoss: {Train_GLoss:.4f}')

    with torch.no_grad():
        for x, y in test_dataloader:
            x = x.to(device)
            y = y.to(device)
            # 前向传播
            with torch.cuda.amp.autocast():
                y_fake = gen(x)
                # metrics
                SSIM_value = ssim(y_fake, y, data_range=1.0, size_average=True)
                MAE_value = torch.nn.L1Loss()(y_fake, y)
                PSNR_value = psnr(y_fake, y)

            SSIM_metrics += SSIM_value.item()
            MAE_metrics += MAE_value.item()
            PSNR_metrics += PSNR_value.item()

    SSIM_metrics /= len(test_dataloader)
    MAE_metrics /= len(test_dataloader)
    PSNR_metrics /= len(test_dataloader)

    # loss_history['test_loss'][epoch]=test_loss
    loss_history['Test_SSIM'][epoch] = SSIM_metrics
    loss_history['Test_MAE'][epoch] = MAE_metrics
    loss_history['Test_PSNR'][epoch] = PSNR_metrics

    print(f'Epoch [{epoch + 1}], SSIM: {SSIM_metrics:.4f}, PSNR: {PSNR_metrics:.4f}, MAE: {MAE_metrics:.4f}')

    # save model prediction and state_dict
    image = y_fake[:4].detach()
    GT_image = y[:4].detach()
    image_torch = torch.cat((image, GT_image), 0)

    if not os.path.exists('./results/generated_images/'):
        os.mkdir('./results/generated_images/')

    if not os.path.exists('./results/state_dict/'):
        os.mkdir('./results/state_dict/')

    torchvision.utils.save_image(image_torch, f"./results/generated_images/image_epoch_{str(epoch + 1).rjust(2, '0')}.png", nrow=4)
    torch.save(gen.state_dict(), f"./results/state_dict/epoch_{str(epoch + 1).rjust(2, '0')}.pth")

    return loss_history

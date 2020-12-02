import time

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.utils as vutils

import utils
from CONSTANTS import Constants
from info_GAN_model import Generator, Discriminator, DHead, QHead
from utils import NormalNLLLoss


class GAN_Manager:
    def __init__(self, device, dataset_name, num_z, num_disc_c,
                 disc_c_dim, num_con_c, dataloader):
        self.device = device
        self.dataset_name = dataset_name
        self.num_z = num_z  # 62 for MNIST
        self.num_disc_c = num_disc_c  # 1 for MNIST
        self.disc_c_dim = disc_c_dim  # 10 for MNIST
        self.num_con_c = num_con_c  # 2 for MNIST
        self.dataloader = dataloader

    def train_info_GAN(self):
        train_img_fig_name = Constants.INFO_GAN_TRAIN_IMAGE_PATH.format(self.dataset_name)
        utils.plot_train_images(self.device, self.dataloader, train_img_fig_name)

        netG = Generator().to(self.device)
        netG.apply(utils.weights_init_InfoGAN)
        print(netG)

        discriminator = Discriminator().to(self.device)
        discriminator.apply(utils.weights_init_InfoGAN)
        print(discriminator)

        netD = DHead().to(self.device)
        netD.apply(utils.weights_init_InfoGAN)
        print(netD)

        netQ = QHead().to(self.device)
        netQ.apply(utils.weights_init_InfoGAN)
        print(netQ)

        # List variables to store results pf training.
        img_list = []
        G_losses = []
        D_losses = []

        optimD = optim.Adam([{'params': discriminator.parameters()}, {'params': netD.parameters()}],
                            lr=Constants.INFO_GAN_LR, betas=(0.5, 0.999))
        optimG = optim.Adam([{'params': netG.parameters()}, {'params': netQ.parameters()}],
                            lr=Constants.INFO_GAN_LR, betas=(0.5, 0.999))

        # Loss for discrimination between real and fake images.
        criterion_D = nn.BCELoss()
        # Loss for discrete latent code.
        criterion_Q_dis = nn.CrossEntropyLoss()
        # Loss for continuous latent code.
        criterion_Q_con = NormalNLLLoss()

        start_time = time.time()
        iters = 0
        real = 1.
        fake = 0.
        fixed_noise = None
        for epoch in range(Constants.INFO_GAN_EPOCH):
            epoch_start_time = time.time()

            for i, (data, _) in enumerate(self.dataloader, 0):
                # get bach size
                # print(data.size())
                b_size = data.size(0)
                # print(b_size)

                real_data = data.to(self.device)

                # Updating discriminator and DHead
                optimD.zero_grad()

                # real data
                label_real = torch.full((b_size,), real, device=self.device)
                # print(label_real)
                output1 = discriminator(real_data)
                probs_real = netD(output1).view(-1)
                loss_real = criterion_D(probs_real, label_real)
                # calculate discriminator grad real data
                loss_real.backward()

                # fake data
                label_fake = torch.full((b_size,), fake, device=self.device)
                noise, idx = utils.noise_sample(self.num_disc_c,
                                                self.disc_c_dim,
                                                self.num_con_c,
                                                self.num_z,
                                                b_size,
                                                self.device)
                fixed_noise = noise

                fake_data = netG(noise)
                output2 = discriminator(fake_data.detach())
                probs_fake = netD(output2).view(-1)
                loss_fake = criterion_D(probs_fake, label_fake)
                # calculate discriminator grad real data
                loss_fake.backward()

                # Net Loss for the discriminator
                D_loss = loss_real + loss_fake
                # Update parameters
                optimD.step()

                # Updating Generator and QHead
                optimG.zero_grad()
                # Fake data treated as real
                output = discriminator(fake_data)
                label_real.fill_(real)
                probs_fake = netD(output).view(-1)
                gen_loss = criterion_D(probs_fake, label_real)

                q_logits, q_mu, q_var = netQ(output)
                target = torch.LongTensor(idx).to(self.device)

                # Calculating loss for discrete latent code.
                info_disc_loss = 0
                for j in range(self.num_disc_c):
                    info_disc_loss += criterion_Q_dis(q_logits[:, j * 10: j * 10 + 10], target[j])

                # Calculating loss for continuous latent code.
                info_con_loss = 0
                if self.num_con_c != 0:
                    info_con_loss += criterion_Q_con(noise[:,
                                                     self.num_z + self.num_disc_c * self.disc_c_dim:]
                                                     .view(-1, self.num_con_c), q_mu, q_var) * 0.1

                    # Net loss for generator.
                    G_loss = gen_loss + info_disc_loss + info_con_loss
                    # Calculate gradients.
                    G_loss.backward()
                    # Update parameters.
                    optimG.step()

                    # Check progress of training.
                    if i != 0 and i % 100 == 0:
                        print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f'
                              % (epoch + 1, Constants.INFO_GAN_EPOCH, i, len(self.dataloader),
                                 D_loss.item(), G_loss.item()))

                    # Save the losses for plotting.
                    G_losses.append(G_loss.item())
                    D_losses.append(D_loss.item())

                    iters += 1

            epoch_time = time.time() - epoch_start_time
            print("Time taken for Epoch %d: %.2fs" % (epoch + 1, epoch_time))
            # Generate image after each epoch to check performance of the generator. Used for creating animated gif later.
            with torch.no_grad():
                gen_data = netG(fixed_noise).detach().cpu()
            img_list.append(vutils.make_grid(gen_data, nrow=10, padding=2, normalize=True))
            utils.save_img_gen_each_epoch(epoch, Constants.INFO_GAN_EPOCH, netG,
                                          fixed_noise, self.dataset_name)

            # Save network weights.
            if (epoch+1) % Constants.INFO_GAN_EPOCH  == 0:
                torch.save({
                    'netG': netG.state_dict(),
                    'discriminator': discriminator.state_dict(),
                    'netD': netD.state_dict(),
                    'netQ': netQ.state_dict(),
                    'optimD': optimD.state_dict(),
                    'optimG': optimG.state_dict(),
                }, 'checkpoint/Info_GAN/model_epoch_%d_{}.pt'.format(self.dataset_name) % (epoch + 1))

        training_time = time.time() - start_time
        print("-" * 50)
        print('Training finished!\nTotal Time for Training: %.2fm' % (training_time / 60))
        print("-" * 50)

        # Generate image to check performance of trained generator.
        with torch.no_grad():
            gen_data = netG(fixed_noise).detach().cpu()
        plt.figure(figsize=(10, 10))
        plt.axis("off")
        plt.imshow(np.transpose(vutils.make_grid(gen_data, nrow=10, padding=2, normalize=True), (1, 2, 0)))
        plt.savefig(Constants.INFO_GAN_TRAIN_IMAGE_PATH_AFTER_TRAINING.
                    format(self.dataset_name) % (Constants.INFO_GAN_EPOCH))

        # Save network weights.
        torch.save({
            'netG': netG.state_dict(),
            'discriminator': discriminator.state_dict(),
            'netD': netD.state_dict(),
            'netQ': netQ.state_dict(),
            'optimD': optimD.state_dict(),
            'optimG': optimG.state_dict(),
        }, 'checkpoint/INFO_GAN/INFO_GAN_FINAL_MODEL_{}.pt'.format(self.dataset_name))

        utils.plot_loss_GAN(G_losses, D_losses, self.dataset_name)
        utils.plot_animation(img_list, self.dataset_name)

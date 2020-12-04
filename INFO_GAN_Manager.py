import time

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.utils as vutils
import matplotlib.animation as animation
import utils
from CONSTANTS import Constants
from info_GAN_model_MNIST import Generator, Discriminator, DHead, QHead


class INFO_GAN_Manager:
    def __init__(self, device, dataset_name, num_z, num_disc_c,
                 disc_c_dim, num_con_c, dataloader):
        self.device = device
        self.dataset_name = dataset_name
        self.num_z = num_z  # 62 for MNIST
        self.num_dis_c = num_disc_c  # 1 for MNIST
        self.dis_c_dim = disc_c_dim  # 10 for MNIST
        self.num_con_c = num_con_c  # 2 for MNIST
        self.dataloader = dataloader

    def train_info_GAN(self):
        # Plot the training images.
        utils.plot_train_images(self.device, self.dataloader,
                                'Info_GAN_Training Images {}'.format(self.dataset_name))

        # Initialise the network.
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

        # Loss for discrimination between real and fake images.
        criterionD = nn.BCELoss()
        # Loss for discrete latent code.
        criterionQ_dis = nn.CrossEntropyLoss()
        # Loss for continuous latent code.
        criterionQ_con = utils.NormalNLLLoss()

        # Adam optimiser is used.
        optimD = optim.Adam(
            [
                {'params': discriminator.parameters()},
                {'params': netD.parameters()}
            ],
            lr=Constants.INFO_GAN_LR,
            betas=(0.5, 0.999)
        )
        optimG = optim.Adam(
            [
                {'params': netG.parameters()},
                {'params': netQ.parameters()}
            ],
            lr=Constants.INFO_GAN_LR,
            betas=(0.5, 0.999))

        # Fixed Noise
        z = torch.randn(100, self.num_z, 1, 1, device=self.device)
        fixed_noise = z
        if self.num_dis_c != 0:
            idx = np.arange(self.dis_c_dim).repeat(10)
            dis_c = torch.zeros(100, self.num_dis_c, self.dis_c_dim, device=self.device)
            for i in range(self.num_dis_c):
                dis_c[torch.arange(0, 100), i, idx] = 1.0

            dis_c = dis_c.view(100, -1, 1, 1)

            fixed_noise = torch.cat((fixed_noise, dis_c), dim=1)

        if self.num_con_c != 0:
            con_c = torch.rand(100, self.num_con_c, 1, 1, device=self.device) * 2 - 1
            fixed_noise = torch.cat((fixed_noise, con_c), dim=1)

        real_label = 1.
        fake_label = 0.

        # List variables to store results pf training.
        img_list = []
        G_losses = []
        D_losses = []

        print("-" * 25)
        print("Starting Training Loop...\n")
        print('Epochs: %d\nDataset: {}\nBatch Size: %d\nLength of Data Loader: %d'.format(self.dataset_name) % (
            Constants.INFO_GAN_EPOCH, 128, len(self.dataloader)))
        print("-" * 25)

        start_time = time.time()
        iters = 0

        for epoch in range(Constants.INFO_GAN_EPOCH):
            epoch_start_time = time.time()

            for i, (data, _) in enumerate(self.dataloader, 0):
                # Get batch size
                b_size = data.size(0)
                # Transfer data tensor to GPU/CPU (device)
                real_data = data.to(self.device)

                # Updating discriminator and DHead
                optimD.zero_grad()
                # Real data
                label = torch.full((b_size,), real_label, device=self.device)
                output1 = discriminator(real_data)
                probs_real = netD(output1).view(-1)
                loss_real = criterionD(probs_real, label)
                # Calculate gradients.
                loss_real.backward()

                # Fake data
                label.fill_(fake_label)
                noise, idx = utils.noise_sample(self.num_dis_c, self.dis_c_dim, self.num_con_c,
                                                self.num_z, b_size, self.device)
                fake_data = netG(noise)
                output2 = discriminator(fake_data.detach())
                probs_fake = netD(output2).view(-1)
                loss_fake = criterionD(probs_fake, label)
                # Calculate gradients.
                loss_fake.backward()

                # Net Loss for the discriminator
                D_loss = loss_real + loss_fake
                # Update parameters
                optimD.step()

                # Updating Generator and QHead
                optimG.zero_grad()

                # Fake data treated as real.
                output = discriminator(fake_data)
                label.fill_(real_label)
                probs_fake = netD(output).view(-1)
                gen_loss = criterionD(probs_fake, label)

                q_logits, q_mu, q_var = netQ(output)
                target = torch.LongTensor(idx).to(self.device)
                # Calculating loss for discrete latent code.
                dis_loss = 0
                for j in range(self.num_dis_c):
                    dis_loss += criterionQ_dis(q_logits[:, j * 10: j * 10 + 10], target[j])

                # Calculating loss for continuous latent code.
                con_loss = 0
                if self.num_con_c != 0:
                    con_loss = criterionQ_con(
                        noise[:, self.num_z + self.num_dis_c * self.dis_c_dim:]
                            .view(-1, self.num_con_c), q_mu, q_var) * 0.1

                # Net loss for generator.
                G_loss = gen_loss + dis_loss + con_loss
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
            # Generate image after each epoch to check performance of the generator.
            # Used for creating animated gif later.
            with torch.no_grad():
                gen_data = netG(fixed_noise).detach().cpu()
            img_list.append(vutils.make_grid(gen_data, nrow=10, padding=2, normalize=True))

            # Generate image to check performance of generator.
            if (epoch + 1) == 1 or (epoch + 1) == Constants.INFO_GAN_EPOCH / 2:
                with torch.no_grad():
                    gen_data = netG(fixed_noise).detach().cpu()
                plt.figure(figsize=(10, 10))
                plt.axis("off")
                plt.imshow(np.transpose(vutils.make_grid(gen_data,
                                                         nrow=10, padding=2, normalize=True), (1, 2, 0)))
                plt.savefig("Epoch_%d {}".format(self.dataset_name) % (epoch + 1))
                plt.close('all')

            # Save network weights.
            if (epoch + 1) % 25 == 0:
                torch.save({
                    'netG': netG.state_dict(),
                    'discriminator': discriminator.state_dict(),
                    'netD': netD.state_dict(),
                    'netQ': netQ.state_dict(),
                    'optimD': optimD.state_dict(),
                    'optimG': optimG.state_dict(),
                    'params': Constants.params
                }, 'checkpoint/model_epoch_%d_{}'.format(self.dataset_name) % (epoch + 1))


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
        plt.savefig("Epoch_%d_{}".format(self.dataset_name) % (Constants.INFO_GAN_EPOCH))

        # Save network weights.
        torch.save({
            'netG': netG.state_dict(),
            'discriminator': discriminator.state_dict(),
            'netD': netD.state_dict(),
            'netQ': netQ.state_dict(),
            'optimD': optimD.state_dict(),
            'optimG': optimG.state_dict(),
            'params': Constants
        }, 'checkpoint/model_final_{}'.format(self.dataset_name))

        # Plot the training losses.
        plt.figure(figsize=(10, 5))
        plt.title("Generator and Discriminator Loss During Training")
        plt.plot(G_losses, label="G")
        plt.plot(D_losses, label="D")
        plt.xlabel("iterations")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig("Loss Curve {}".format(self.dataset_name))

        # Animation showing the improvements of the generator.
        fig = plt.figure(figsize=(10, 10))
        plt.axis("off")
        ims = [[plt.imshow(np.transpose(i, (1, 2, 0)), animated=True)] for i in img_list]
        anim = animation.ArtistAnimation(fig, ims, interval=1000, repeat_delay=1000, blit=True)
        anim.save('infoGAN_{}.gif'.format(self.dataset_name), dpi=80, writer='imagemagick')
        plt.show()


    def generate_images(self):
        state_dict = torch.load("./checkpoint/InfoGAN/MNIST/model_final_MNIST",
                                map_location=torch.device('cpu'))

        # Set the device to run on: GPU or CPU.
        device = torch.device("cuda:0" if (torch.cuda.is_available())
                              else "cpu")
        # Get the 'params' dictionary from the loaded state_dict.
        params = state_dict['params']

        # Create the generator network.
        netG = Generator().to(device)
        # Load the trained generator weights.
        netG.load_state_dict(state_dict['netG'])
        print(netG)

        c = np.linspace(-2, 2, 10).reshape(1, -1)
        c = np.repeat(c, 3000, 0).reshape(-1, 1)
        # print(c.shape)
        # print(c)
        c = torch.from_numpy(c).float().to(device)
        c = c.view(-1, 1, 1, 1)
        # print(c.size())
        zeros = torch.zeros(30000, 1, 1, 1, device=device)

        # Continuous latent code.
        c2 = torch.cat((c, zeros), dim=1)
        c3 = torch.cat((zeros, c), dim=1)

        idx = np.arange(10).repeat(3000)
        print(idx)
        dis_c = torch.zeros(30000, 10, 1, 1, device=device)
        dis_c[torch.arange(0, 30000), idx] = 1.0

        # Discrete latent code.
        c1 = dis_c.view(30000, -1, 1, 1)

        z = torch.randn(30000, 62, 1, 1, device=device)

        # To see variation along c2 (Horizontally) and c1 (Vertically)
        noise1 = torch.cat((z, c1, c2), dim=1)
        # To see variation along c3 (Horizontally) and c1 (Vertically)
        noise2 = torch.cat((z, c1, c3), dim=1)

        # Generate image.
        with torch.no_grad():
            generated_img1 = netG(noise1).detach().cpu()
        print(generated_img1.size())
        torch.save(generated_img1, '30k_image_set_MNIST_noise_1.pt')

        # Display the generated image.
        fig = plt.figure(figsize=(10, 10))
        plt.axis("off")
        plt.imshow(np.transpose(vutils.make_grid(generated_img1, nrow=10, padding=2, normalize=True), (1, 2, 0)))
        plt.show()

        # Generate image.
        with torch.no_grad():
            generated_img2 = netG(noise2).detach().cpu()


        torch.save(generated_img2, '30k_image_set_MNIST_noise_2.pt')
        # # Display the generated image.
        fig = plt.figure(figsize=(10, 10))
        plt.axis("off")
        plt.imshow(np.transpose(vutils.make_grid(generated_img2, nrow=10, padding=2, normalize=True), (1, 2, 0)))
        plt.show()

        img_tensor = torch.load('tensor.pt')
        print(img_tensor)

        for i in range(1):
            plt.axis('off')
            np_image = np.transpose(generated_img2[i], (1, 2, 0))
            print(np_image.shape)
            plt.imshow(np_image)
            plt.savefig("Image_{0}".format(i))



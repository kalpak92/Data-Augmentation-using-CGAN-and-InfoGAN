import time
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
import torchvision.utils as vutils
from CGAN_model_MNIST import Generator, Discriminator, get_noise
from CONSTANTS import Constants


class CGANManager:
    def __init__(self, device, dataset_name, dataloader, n_classes, z_dim):
        self.device = device
        self.dataset_name = dataset_name
        self.dataloader = dataloader
        self.n_classes = n_classes
        self.z_dim = z_dim

    def get_one_hot_labels(self, labels, n_classes):
        """
        Function for creating one-hot vectors for the labels, returns a tensor of shape (?, num_classes).
        Parameters:
            labels: tensor of labels from the dataloader, size (?)
            n_classes: the total number of classes in the dataset, an integer scalar
        """
        return F.one_hot(labels, n_classes)

    def combine_vectors(self, x, y):
        """
        Function for combining two vectors with shapes (n_samples, ?) and (n_samples, ?).
        Parameters:
          x: (n_samples, ?) the first vector.
            In this assignment, this will be the noise vector of shape (n_samples, z_dim),
            but you shouldn't need to know the second dimension's size.
          y: (n_samples, ?) the second vector.
            Once again, in this assignment this will be the one-hot class vector
            with the shape (n_samples, n_classes), but you shouldn't assume this in your code.
        """

        # Note: Make sure this function outputs a float no matter what inputs it receives
        combined = torch.cat((x.float(), y.float()), 1)
        return combined

    def get_input_dimensions(self, z_dim, mnist_shape, n_classes):
        """
        Function for getting the size of the conditional input dimensions
        from z_dim, the image shape, and number of classes.
        Parameters:
            z_dim: the dimension of the noise vector, a scalar
            mnist_shape: the shape of each MNIST image as (C, W, H), which is (1, 28, 28)
            n_classes: the total number of classes in the dataset, an integer scalar
                    (10 for MNIST)
        Returns:
            generator_input_dim: the input dimensionality of the conditional generator,
                              which takes the noise and class vectors
            discriminator_im_chan: the number of input channels to the discriminator
                                (e.g. C x 28 x 28 for MNIST)
        """
        generator_input_dim = z_dim + n_classes
        discriminator_im_chan = mnist_shape[0] + n_classes
        return generator_input_dim, discriminator_im_chan

    def weights_init(self, m):
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
            torch.nn.init.normal_(m.weight, 0.0, 0.02)
        if isinstance(m, nn.BatchNorm2d):
            torch.nn.init.normal_(m.weight, 0.0, 0.02)
            torch.nn.init.constant_(m.bias, 0)

    def train_CGAN(self):
        mnist_shape = (1, 28, 28)
        generator_input_dim, discriminator_im_chan = self.get_input_dimensions(self.z_dim, mnist_shape,
                                                                               self.n_classes)
        gen = Generator(input_dim=generator_input_dim).to(self.device)
        gen_opt = torch.optim.Adam(gen.parameters(), lr=Constants.C_GAN_LR)
        disc = Discriminator(im_chan=discriminator_im_chan).to(self.device)
        disc_opt = torch.optim.Adam(disc.parameters(), lr=Constants.C_GAN_LR)

        gen = gen.apply(self.weights_init)
        disc = disc.apply(self.weights_init)

        cur_step = 0
        generator_losses = []
        discriminator_losses = []
        criterion = nn.BCEWithLogitsLoss()

        noise_and_labels = False
        fake = False

        fake_image_and_labels = False
        real_image_and_labels = False
        disc_fake_pred = False
        disc_real_pred = False
        start_time = time.time()
        for epoch in range(Constants.CGAN_EPOCH):
            epoch_start_time = time.time()
            # Dataloader returns the batches and the labels
            for i, (real, labels) in enumerate(self.dataloader):
                cur_batch_size = len(real)
                # Flatten the batch of real images from the dataset
                real = real.to(self.device)

                one_hot_labels = self.get_one_hot_labels(labels.to(self.device), self.n_classes)
                image_one_hot_labels = one_hot_labels[:, :, None, None]
                image_one_hot_labels = image_one_hot_labels.repeat(1, 1, mnist_shape[1], mnist_shape[2])

                ### Update discriminator ###
                # Zero out the discriminator gradients
                disc_opt.zero_grad()
                # Get noise corresponding to the current batch_size
                fake_noise = get_noise(cur_batch_size, self.z_dim, device=self.device)

                # Now you can get the images from the generator
                # Steps: 1) Combine the noise vectors and the one-hot labels for the generator
                #        2) Generate the conditioned fake images

                noise_and_labels = self.combine_vectors(fake_noise, one_hot_labels)
                fake = gen(noise_and_labels)

                # Make sure that enough images were generated
                assert len(fake) == len(real)
                # Check that correct tensors were combined
                assert tuple(noise_and_labels.shape) == (cur_batch_size, fake_noise.shape[1] + one_hot_labels.shape[1])
                # It comes from the correct generator
                assert tuple(fake.shape) == (len(real), 1, 28, 28)

                fake_image_and_labels = self.combine_vectors(fake, image_one_hot_labels)
                real_image_and_labels = self.combine_vectors(real, image_one_hot_labels)
                disc_fake_pred = disc(fake_image_and_labels.detach())
                disc_real_pred = disc(real_image_and_labels)

                disc_fake_loss = criterion(disc_fake_pred, torch.zeros_like(disc_fake_pred))
                disc_real_loss = criterion(disc_real_pred, torch.ones_like(disc_real_pred))
                disc_loss = (disc_fake_loss + disc_real_loss) / 2
                disc_loss.backward(retain_graph=True)
                disc_opt.step()

                # Keep track of the average discriminator loss
                discriminator_losses += [disc_loss.item()]

                ### Update generator ###
                # Zero out the generator gradients
                gen_opt.zero_grad()

                fake_image_and_labels = self.combine_vectors(fake, image_one_hot_labels)
                # This will error if you didn't concatenate your labels to your image correctly
                disc_fake_pred = disc(fake_image_and_labels)
                gen_loss = criterion(disc_fake_pred, torch.ones_like(disc_fake_pred))
                gen_loss.backward()
                gen_opt.step()

                # Keep track of the generator losses
                generator_losses += [gen_loss.item()]

                # Check progress of training.
                if i != 0 and i % 100 == 0:
                    print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f'
                          % (epoch + 1, Constants.CGAN_EPOCH, i, len(self.dataloader),
                             disc_loss.item(), gen_loss.item()))


            epoch_time = time.time() - epoch_start_time
            print("Time taken for Epoch %d: %.2fs" % (epoch + 1, epoch_time))

        training_time = time.time() - start_time
        print("-" * 50)
        print('Training finished!\nTotal Time for Training: %.2fm' % (training_time / 60))
        print("-" * 50)

        # Generate image to check performance of trained generator.
        with torch.no_grad():
            one_hot_labels = self.get_one_hot_labels(torch.Tensor([5]).long(), self.n_classes).to(self.device)
            fake_noise = get_noise(1, self.z_dim, device=self.device)
            noise_and_labels = self.combine_vectors(fake_noise, one_hot_labels)
            gen_data = gen(noise_and_labels).detach().cpu()

        plt.figure(figsize=(10, 10))
        plt.axis("off")
        plt.imshow(np.transpose(vutils.make_grid(gen_data, nrow=10, padding=2, normalize=True), (1, 2, 0)))
        plt.savefig(Constants.C_GAN_TRAIN_IMAGE_PATH_AFTER_TRAINING.
                    format(self.dataset_name) % (Constants.CGAN_EPOCH))

        # Save network weights.
        torch.save({
            'Generator': gen.state_dict(),
            'discriminator': disc.state_dict(),
            'optimD': disc_opt.state_dict(),
            'optimG': gen_opt.state_dict(),
        }, 'checkpoint/C_GAN/C_GAN_FINAL_MODEL_{}'.format(self.dataset_name))

        # utils.plot_loss_GAN(G_losses, D_losses, self.dataset_name)
        # utils.plot_animation(img_list, self.dataset_name)
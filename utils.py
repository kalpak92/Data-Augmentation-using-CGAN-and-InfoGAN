import json
import os

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torchvision.utils as vutils

from CONSTANTS import Constants


class configParams():
    def __init__(self, json_path):
        with open(json_path) as f:
            params = json.load(f)
            self.__dict__.update(params)

    def save(self, json_path):
        with open(json_path, 'w') as f:
            json.dump(self.__dict__, f, indent=4)

    def update(self, json_path):
        with open(json_path) as f:
            params = json.load(f)
            self.__dict__.update(params)

    @property
    def dict(self):
        """Gives dict-like access to Params instance by `params.dict['learning_rate']"""
        return self.__dict__


class RunningAverageLoss():
    """A simple class that maintains the running average of a quantity

    Example:
    ```
    loss_avg = RunningAverage()
    loss_avg.update(2)
    loss_avg.update(4)
    loss_avg() = 3
    ```
    """

    def __init__(self):
        self.steps = 0
        self.total = 0

    def update(self, val):
        self.total += val
        self.steps += 1

    def __call__(self):
        return self.total / float(self.steps)


def save_dict_to_json(d, json_path):
    """Saves dict of floats in json file
    Args:
        d: (dict) of float-castable values (np.float, int, float, etc.)
        json_path: (string) path to json file
    """
    with open(json_path, 'w') as f:
        # We need to convert the values to float for json (it doesn't accept np.array, np.float, )
        d = {k: float(v) for k, v in d.items()}
        json.dump(d, f, indent=4)


def save_checkpoint(state, checkpoint):
    """Saves model and training parameters at checkpoint + 'last.pth.tar'. If is_best==True, also saves
    checkpoint + 'best.pth.tar'
    Args:
        state: (dict) contains model's state_dict, may contain other keys such as epoch, optimizer state_dict
        is_best: (bool) True if it is the best model seen till now
        checkpoint: (string) folder where parameters are to be saved
    """
    filepath = os.path.join(checkpoint, 'last.pth.tar')
    if not os.path.exists(checkpoint):
        print("Checkpoint Directory does not exist! Making directory {}".format(checkpoint))
        os.mkdir(checkpoint)
    else:
        print("Checkpoint Directory exists! ")

    torch.save(state, filepath)


def get_device():
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def plot_loss_epoch(train_loss_avg, fig_name):
    plt.ion()
    fig = plt.figure()
    plt.plot(train_loss_avg)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    # plt.show()
    plt.draw()
    plt.savefig(fig_name, dpi=220)
    plt.clf()


def plot_train_images(device, dataloader, fig_name):
    sample_batch = next(iter(dataloader))
    plt.figure(figsize=(10, 10))
    plt.axis("off")
    plt.imshow(np.transpose(vutils.make_grid(
        sample_batch[0].to(device)[: 100], nrow=10, padding=2, normalize=True).cpu(), (1, 2, 0)))
    plt.savefig(fig_name, dpi=220)
    plt.close('all')


def weights_init_InfoGAN(m):
    """
    Initialise weights of the model.
    """
    if (type(m) == nn.ConvTranspose2d or type(m) == nn.Conv2d):
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif (type(m) == nn.BatchNorm2d):
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


class NormalNLLLoss:
    """
    Calculate the negative log likelihood
    of normal distribution.
    This needs to be minimised.
    Treating Q(cj | x) as a factored Gaussian.
    """

    def __call__(self, x, mu, var):
        logli = -0.5 * (var.mul(2 * np.pi) + 1e-6).log() - (x - mu).pow(2).div(var.mul(2.0) + 1e-6)
        nll = -(logli.sum(1).mean())

        return nll


def noise_sample(n_dis_c, dis_c_dim, n_con_c, n_z, batch_size, device):
    """
    Sample random noise vector for training.
    INPUT
    --------
    n_dis_c : Number of discrete latent code.
    dis_c_dim : Dimension of discrete latent code.
    n_con_c : Number of continuous latent code.
    n_z : Dimension of iicompressible noise.
    batch_size : Batch Size
    device : GPU/CPU
    """

    dis_c = None
    con_c = None
    z = torch.randn(batch_size, n_z, 1, 1, device=device)

    idx = np.zeros((n_dis_c, batch_size))
    if n_dis_c != 0:
        dis_c = torch.zeros(batch_size, n_dis_c, dis_c_dim, device=device)

        for i in range(n_dis_c):
            idx[i] = np.random.randint(dis_c_dim, size=batch_size)
            dis_c[torch.arange(0, batch_size), i, idx[i]] = 1.0

        dis_c = dis_c.view(batch_size, -1, 1, 1)

    if n_con_c != 0:
        # Random uniform between -1 and 1.
        con_c = torch.rand(batch_size, n_con_c, 1, 1, device=device) * 2 - 1

    noise = z
    if n_dis_c != 0:
        noise = torch.cat((z, dis_c), dim=1)
    if n_con_c != 0:
        noise = torch.cat((noise, con_c), dim=1)
    # print("fixed_noise (disc_c + cont_c): ", noise.shape)

    return noise, idx


def save_img_gen_each_epoch(epoch_num, total_epoch, netG, fixed_noise, dataset_name):
    # Generate image to check performance of generator.
    if (epoch_num + 1) == 1 or (epoch_num + 1) == total_epoch / 2:
        with torch.no_grad():
            gen_data = netG(fixed_noise).detach().cpu()
        plt.figure(figsize=(10, 10))
        plt.axis("off")
        plt.imshow(np.transpose(vutils.make_grid(gen_data, nrow=10, padding=2, normalize=True), (1, 2, 0)))
        plt.savefig(Constants.INFO_GAN_TRAIN_IMAGE_PER_EPOCH_PATH +
                    "/Epoch_%d {}".format(dataset_name) % (epoch_num + 1))
        plt.close('all')


def plot_loss_GAN(G_losses, D_losses, dataset_name):
    # Plot the training losses.
    plt.figure(figsize=(10, 5))
    plt.title("Generator and Discriminator Loss During Training")
    plt.plot(G_losses, label="G")
    plt.plot(D_losses, label="D")
    plt.xlabel("iterations")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(Constants.INFO_GAN_LOSS_PATH.format(dataset_name))


def plot_animation(img_list, dataset_name):
    # Animation showing the improvements of the generator
    fig = plt.figure(figsize=(10, 10))
    plt.axis("off")
    ims = [[plt.imshow(np.transpose(i, (1, 2, 0)), animated=True)]
           for i in img_list]
    anim = animation.ArtistAnimation(fig, ims, interval=1000,
                                     repeat_delay=1000, blit=True)
    anim.save(Constants.INFO_GAN_ANIM_PATH.format(dataset_name), dpi=80,
              writer='imagemagick')
    plt.show()

import torch
import utils
from CONSTANTS import Constants
from CGAN_manager import CGANManager
from INFO_GAN_Manager import INFO_GAN_Manager
from build_dataset import DataLoader


class GAN_experiments:
    @staticmethod
    def info_GAN_Experiments(MNIST_params, device):
        dl_mnist = DataLoader(MNIST_params["database_name"])
        train_dataloader_mnist = dl_mnist.get_train_val_dataloader(batch_size=128, shuffle=True)
        info_gan = INFO_GAN_Manager(device, MNIST_params["database_name"],
                                    MNIST_params["MNIST_num_z"],
                                    MNIST_params["MNIST_num_disc_c"],
                                    MNIST_params["MNIST_disc_c_dim"],
                                    MNIST_params["MNIST_num_con_c"],
                                    train_dataloader_mnist)
        info_gan.train_info_GAN()

    @staticmethod
    def c_GAN_Experiments(CGAN_params, device):
        dl_mnist = DataLoader(CGAN_params["database_name"])
        if torch.cuda.is_available():
            train_dataloader_mnist = dl_mnist.get_train_val_dataloader(batch_size=128,
                                                                       num_workers=4, pin_memory=True)
        else:
            train_dataloader_mnist = dl_mnist.get_train_val_dataloader(batch_size=128)
        c_gan = CGANManager(device, CGAN_params["database_name"], train_dataloader_mnist,
                            CGAN_params["n_classes"], CGAN_params["z_dim"])
        c_gan.train_CGAN()


MNIST_params = {
    "database_name": Constants.MNIST,
    "MNIST_num_z": Constants.MNIST_num_z,
    "MNIST_num_disc_c": Constants.MNIST_num_disc_c,
    "MNIST_disc_c_dim": Constants.MNIST_disc_c_dim,
    "MNIST_num_con_c": Constants.MNIST_num_con_c
}
device = utils.get_device()

CGAN_params = {
    "database_name": Constants.MNIST,
    "n_classes" : Constants.CGAN_N_CLASSES,
    "z_dim" : Constants.CGAN_Z_DIM
}

# info_GAN = GAN_experiments()
# info_GAN.info_GAN_Experiments(MNIST_params, device)

c_GAN = GAN_experiments()
c_GAN.c_GAN_Experiments(CGAN_params, device)




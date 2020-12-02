import utils
from CONSTANTS import Constants
from GAN_Manager import GAN_Manager
from build_dataset import DataLoader


class GAN_experiments:
    @staticmethod
    def info_GAN_Experiments(MNIST_params, device):
        dl_mnist = DataLoader(MNIST_params["database_name"])
        train_dataloader_mnist = dl_mnist.get_train_val_dataloader(batch_size=128, shuffle=True)
        info_gan = GAN_Manager(device, MNIST_params["database_name"],
                               MNIST_params["MNIST_num_z"],
                               MNIST_params["MNIST_num_disc_c"],
                               MNIST_params["MNIST_disc_c_dim"],
                               MNIST_params["MNIST_num_con_c"],
                               train_dataloader_mnist)
        info_gan.train_info_GAN()


MNIST_params = {
    "database_name": Constants.MNIST,
    "MNIST_num_z": Constants.MNIST_num_z,
    "MNIST_num_disc_c": Constants.MNIST_num_disc_c,
    "MNIST_disc_c_dim": Constants.MNIST_disc_c_dim,
    "MNIST_num_con_c": Constants.MNIST_num_con_c
}
device = utils.get_device()

info_GAN = GAN_experiments()
info_GAN.info_GAN_Experiments(MNIST_params, device)

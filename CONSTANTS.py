class Constants:
    MNIST = "MNIST"
    MNIST_num_z = 62
    MNIST_num_disc_c = 1
    MNIST_disc_c_dim = 10
    MNIST_num_con_c = 2

    INFO_GAN_TRAIN_IMAGE_PATH = "./Info_GAN_Images/Training_Images_before_Training_{}"
    INFO_GAN_TRAIN_IMAGE_PATH_AFTER_TRAINING = "./Info_GAN_Images/Training_Images_after_Training_%d_{}"
    INFO_GAN_ANIM_PATH = "./Info_GAN_Images/infoGAN_{}.gif"

    INFO_GAN_LOSS_PATH = "./Info_GAN_Images/Loss Curve {}"
    INFO_GAN_TRAIN_IMAGE_PER_EPOCH_PATH = "./Info_GAN_Images/Generated_image_per_epoch"
    INFO_GAN_LR = 2e-4
    INFO_GAN_EPOCH = 3000

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
    INFO_GAN_EPOCH = 100

    C_GAN_TRAIN_IMAGE_PATH = "./C_GAN_Images/Training_Images_before_Training_{}"
    C_GAN_TRAIN_IMAGE_PATH_AFTER_TRAINING = "./C_GAN_Images/Training_Images_after_Training_%d_{}"
    C_GAN_ANIM_PATH = "./C_GAN_Images/infoGAN_{}.gif"

    CGAN_N_CLASSES = 10
    CGAN_Z_DIM = 64
    C_GAN_LR = 0.0002
    C_GAN_DISPLAY_STEP = 500
    C_GAN_BATCH_SIZE = 128
    params = {
        'batch_size': 128,  # Batch size.
        'num_epochs': 100,  # Number of epochs to train for.
        'learning_rate': 2e-4,  # Learning rate.
        'beta1': 0.5,
        'beta2': 0.999,
        'save_epoch': 25,  # After how many epochs to save checkpoints and generate test output.
        'dataset': 'MNIST'
    }
    CGAN_EPOCH = 500

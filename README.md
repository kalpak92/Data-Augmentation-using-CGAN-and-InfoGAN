# Data Augmentation using Conditional GAN and InfoGAN

## Program Execution

Set the path of **`model_dir`** in  **`convNet_manager.py`** corresponding to the folder present under **experiments** directory depending upon the configuration of your choice.

For example : `experiments/mnist_infoGan_30k_noise1` has the configurations detailed under **`config.json`** and set **`model_dir='experiments/mnist_infoGan_30k_noise1'`** in **`convNet_manager.py`**.

Then, execute the command:

```shell
python3 convNet_manager.py
```



## Folder Structure

```shell
├── CGAN_manager.py
├── CGAN_model_MNIST.py
├── CONSTANTS.py
├── C_GAN_Images
│   ├── Gen_Imgae_final.jpg
│   ├── Training_Images_after_Training_1_MNIST.png
│   └── Training_Images_after_Training_500_MNIST.png
├── C_GAN_generate_datasets
│   ├── 30k_image_set_MNIST_noise_1.pt
│   ├── 48k_image_set_MNIST_noise_1.pt
│   ├── 54k_image_set_MNIST_noise_1.pt
│   └── sample.txt
├── GAN_experiments.py
├── INFO_GAN_Manager.py
├── Info_GAN_Plots
├── Info_GAN_generate_datasets
│   ├── 30k_image_set_MNIST_noise_1.pt
│   ├── 30k_image_set_MNIST_noise_2.pt
│   ├── 48k_image_set_MNIST_noise_1.pt
│   ├── 48k_image_set_MNIST_noise_2.pt
│   ├── 54k_image_set_MNIST_noise_1.pt
│   ├── 54k_image_set_MNIST_noise_2.pt
│   └── sample.txt
├── README.md
├── Rough.py
├── augment_dataset.py
├── build_dataset.py
├── checkpoint
├── convNet.py
├── convNet_manager.py
├── data
├── experiments
│   ├── base_cnn
│   │   ├── config.json
│   │   ├── last.pth.tar
│   │   ├── loss_plot.jpeg
│   │   └── metrics_test.json
│   ├── mnist_cGan_30k
│   │   ├── ConfusionMatrix.jpeg
│   │   ├── config.json
│   │   ├── last.pth.tar
│   │   ├── loss_plot.jpeg
│   │   └── metrics_test.json
│   ├── mnist_cGan_48k
│   │   ├── ConfusionMatrix.jpeg
│   │   ├── config.json
│   │   ├── last.pth.tar
│   │   ├── loss_plot.jpeg
│   │   └── metrics_test.json
│   ├── mnist_cGan_54k
│   │   ├── ConfusionMatrix.jpeg
│   │   ├── config.json
│   │   ├── last.pth.tar
│   │   ├── loss_plot.jpeg
│   │   └── metrics_test.json
│   ├── mnist_infoGan_30k_noise1
│   │   ├── ConfusionMatrix.jpeg
│   │   ├── config.json
│   │   ├── last.pth.tar
│   │   ├── loss_plot.jpeg
│   │   └── metrics_test.json
│   ├── mnist_infoGan_30k_noise2
│   │   ├── ConfusionMatrix.jpeg
│   │   ├── config.json
│   │   ├── last.pth.tar
│   │   ├── loss_plot.jpeg
│   │   └── metrics_test.json
│   ├── mnist_infoGan_48k_noise1
│   │   ├── ConfusionMatrix.jpeg
│   │   ├── config.json
│   │   ├── last.pth.tar
│   │   ├── loss_plot.jpeg
│   │   └── metrics_test.json
│   ├── mnist_infoGan_48k_noise2
│   │   ├── ConfusionMatrix.jpeg
│   │   ├── config.json
│   │   ├── last.pth.tar
│   │   ├── loss_plot.jpeg
│   │   └── metrics_test.json
│   ├── mnist_infoGan_54k_noise1
│   │   ├── ConfusionMatrix.jpeg
│   │   ├── config.json
│   │   ├── last.pth.tar
│   │   ├── loss_plot.jpeg
│   │   └── metrics_test.json
│   └── mnist_infoGan_54k_noise2
│       ├── ConfusionMatrix.jpeg
│       ├── config.json
│       ├── last.pth.tar
│       ├── loss_plot.jpeg
│       └── metrics_test.json
├── info_GAN_model_MNIST.py
├── notebooks
│   ├── Build_a_Conditional_GAN.ipynb
│   ├── InfoGAN.ipynb
│   ├── build_cnn_pytorch.ipynb
│   └── visualize_with_tensorboard.ipynb
├── plot_confusion_matrix.py
├── test.py
├── train.py
└── utils.py

```



## Model Description

The Convolution Neural Network architecture comprises of **convolution**, **max pool** and **batch** **normalization** operations in each layer. Each convolution layer uses a 3x3 filter with padding and stride set to 1. The network comprises of 3 such layers, after which the information is flattened and sent to a fully connected Neural Network having two hidden layers. The fully connected network can have dropout as a regularization parameters set by the user as part of the configuration.



## Experiments

### Network Details

We have used PyTorch to develop the network. 

The weights of the Convolutional Neural Networks are initialised using ***Kaiming*** for the convolutional layers and ***Xavier*** for the fully connected layers. The optimizer used is **Adam** and learning rate is set to `1e-3`, but the user can set any learning rate as part of the configuration. 

### Datasets

We used 

## References

- [Generative Adversarial Networks](https://arxiv.org/pdf/1406.2661.pdf)

- [Conditional Generative Adversarial Networks](https://arxiv.org/pdf/1411.1784.pdf)

- [InfoGAN: Interpretable Representation Learning by Information Maximizing Generative Adversarial Nets](https://arxiv.org/abs/1606.03657)

- [Build Basic Generative Adversarial Networks (GANs) - Deeplearning.ai](https://www.coursera.org/learn/build-basic-generative-adversarial-networks-gans)


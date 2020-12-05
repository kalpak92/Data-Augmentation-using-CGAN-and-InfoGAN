import matplotlib.pyplot as plt
import numpy as np
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torch.utils.data import ConcatDataset

from utils import *

img_tensor = torch.load('30k_image_set_MNIST_noise_1.pt')
idx = np.arange(10).repeat(3000)

# img_tensor = torch.load('48k_image_set_MNIST_noise_1.pt')
# img_tensor = torch.load('54k_image_set_MNIST_noise_2.pt')
print(img_tensor.size())
print(idx.shape)
print(idx[0])

plt.axis('off')
# print(img_tensor[3000].size())

arr_0 = np.empty(3000)
arr_0.fill(3)
print(arr_0.shape)
print(arr_0)

arr_1 = np.empty(3000)
arr_1.fill(4)
print(arr_1.shape)
print(arr_1)

arr_2 = np.empty(3000)
arr_2.fill(6)
print(arr_2.shape)
print(arr_2)

arr_3 = np.empty(3000)
arr_3.fill(7)
print(arr_3.shape)
print(arr_3)

arr_4 = np.empty(3000)
arr_4.fill(2)
print(arr_4.shape)
print(arr_4)

arr_5 = np.empty(3000)
arr_5.fill(1)
print(arr_5.shape)
print(arr_5)

arr_6 = np.empty(3000)
arr_6.fill(5)
print(arr_6.shape)
print(arr_6)

arr_7 = np.empty(3000)
arr_7.fill(9)
print(arr_7.shape)
print(arr_7)

arr_8 = np.empty(3000)
arr_8.fill(0)
print(arr_8.shape)
print(arr_8)

arr_9 = np.empty(3000)
arr_9.fill(8)
print(arr_9.shape)
print(arr_9)

labels_2D = np.concatenate((arr_0, arr_1, arr_2, arr_3, arr_4, arr_5, arr_6, arr_7, arr_8, arr_9), axis=0)
print(labels_2D.shape)
print(labels_2D)
print("-----------")

np_image = np.transpose(img_tensor[27000], (1, 2, 0))
# print(np_image.shape)
plt.imshow(np_image)
plt.show()

#
# for i in range(1, 30000, 2999):
#     print(i)
#     plt.axis('off')
#     print(img_tensor[i].size())
#     np_image = np.transpose(img_tensor[i], (1, 2, 0))
#     # print(np_image.shape)
#     plt.imshow(np_image)
#     plt.show()
#     # plt.savefig("Image_{0}".format(i))

root = 'data/'

transform = transforms.Compose([
    transforms.Resize(28),
    transforms.CenterCrop(28),
    transforms.ToTensor()])

dataset = dsets.MNIST(root + 'mnist/', train='train',
                      download=True, transform=transform)
dataloader = torch.utils.data.DataLoader(dataset,
                                         batch_size=60000,
                                         shuffle=True)
image_tensor_MNIST = None
label_tensor_MNIST = None
for i, (data, label) in enumerate(dataloader, 0):
    image_tensor_MNIST = data
    label_tensor_MNIST = label

tensor_y = torch.from_numpy(labels_2D)

print(image_tensor_MNIST.size())
print(img_tensor.size())
Big_img_tensor = torch.cat((image_tensor_MNIST, img_tensor), 0)

print(label_tensor_MNIST.size())
print(tensor_y.size())
Big_label_tensor = torch.cat((label_tensor_MNIST, tensor_y), 0)

print("Big_tensor ---->>")
print(Big_img_tensor.size())
print(Big_label_tensor.size())

processed_dataset = torch.utils.data.TensorDataset(Big_img_tensor, Big_label_tensor)
print(processed_dataset)

aug_dataloader = torch.utils.data.DataLoader(processed_dataset,
                                         batch_size=128,
                                         shuffle=True)

for i, (data, label) in enumerate(aug_dataloader, 0):
    image_tensor_MNIST = data[0]
    label_tensor_MNIST = label[0]
    plt.axis('off')
    print(image_tensor_MNIST[i].size())
    np_image = np.transpose(img_tensor[i], (1, 2, 0))
    # print(np_image.shape)
    plt.imshow(np_image)
    plt.show()
    break

print(type(processed_dataset))

print(len(aug_dataloader))

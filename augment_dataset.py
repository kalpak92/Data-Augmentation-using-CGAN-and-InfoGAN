from itertools import repeat
import numpy as np
import torch
from torch.utils.data import ConcatDataset
import matplotlib.pyplot as plt
from build_dataset import DataLoader


class AugmentDataset:
    def get_augmented_dataloader(self, original_dataloader=DataLoader("MNIST"), split_percentage=1, batch_size=64,
                                 shuffle=True, num_workers=1, pin_memory=False,
                                 generated_dataset_path='./Info_GAN_generate_datasets/30k_image_set_MNIST_noise_1.pt'):
        generated_image_tensor = torch.load(generated_dataset_path)
        generated_labels = self.get_labels()

        subset_train_datasetA, subset_train_datasetB = \
            original_dataloader.get_train_dataset_split(split_percentage=split_percentage)
        subset_test_dataset = original_dataloader.get_test_dataset_split(split_percentage=split_percentage)

        original_data_tensor, original_label_tensor = self.get_original_image_and_label_tensors(subset_train_datasetA)

        augmented_data_tensor = torch.cat((original_data_tensor, generated_image_tensor), 0)
        augmented_label_tensor = torch.cat((original_label_tensor, generated_labels), 0)
        print("Original data tensor", original_data_tensor.size())
        print("Generated data tensor", generated_image_tensor.size())
        print("original_label_tensor", original_label_tensor.size())
        print("generated_labels", generated_labels.size())
        augmented_dataset = torch.utils.data.TensorDataset(augmented_data_tensor, augmented_label_tensor)
        print("augmented train dataset = ", len(augmented_dataset))
        print("augmented test dataset = ", len(subset_test_dataset))
        augmented_train_dataloader = torch.utils.data.DataLoader(augmented_dataset, batch_size=batch_size,
                                                                 shuffle=shuffle, num_workers=num_workers,
                                                                 pin_memory=pin_memory)

        augmented_test_data_loader = torch.utils.data.DataLoader(
            subset_test_dataset, batch_size=batch_size, shuffle=shuffle,
            num_workers=num_workers, pin_memory=pin_memory)

        return augmented_train_dataloader, augmented_test_data_loader

    def get_original_image_and_label_tensors(self, dataset):
        global data_tensor, label_tensor
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=len(dataset), shuffle=True)
        for i, (data, label) in enumerate(dataloader, 0):
            data_tensor = data
            label_tensor = label

        return data_tensor, label_tensor

    def get_labels(self):
        arr_0 = list(repeat(3, 3000))
        arr_1 = list(repeat(4, 3000))
        arr_2 = list(repeat(6, 3000))
        arr_3 = list(repeat(7, 3000))
        arr_4 = list(repeat(2, 3000))
        arr_5 = list(repeat(1, 3000))
        arr_6 = list(repeat(5, 3000))
        arr_7 = list(repeat(9, 3000))
        arr_8 = list(repeat(0, 3000))
        arr_9 = list(repeat(8, 3000))

        # labels_2D = np.concatenate((arr_0, arr_1, arr_2, arr_3, arr_4, arr_5, arr_6, arr_7, arr_8, arr_9), axis=0)
        # print(labels_2D.shape)
        # print(labels_2D)
        labels_2D = torch.tensor(arr_0 + arr_1 + arr_2 + arr_3 + arr_4 + arr_5 + arr_6 + arr_7 + arr_8 + arr_9)
        return labels_2D

if __name__ == '__main__':
    original_dataloader = DataLoader("MNIST")
    augmented_dataset = AugmentDataset()
    aug_train_dataloader, aug_test_dataloader = augmented_dataset.get_augmented_dataloader(original_dataloader=original_dataloader, split_percentage=0.5)

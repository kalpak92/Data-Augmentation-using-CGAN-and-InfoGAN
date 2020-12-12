import torch
from torch.utils.data import random_split, ConcatDataset
from torchvision import datasets
from torchvision import transforms


class DataLoader:
    def __init__(self, dataset):
        self.dataset_name = dataset
        self.train_dataset = None
        self.test_dataset = None
        self.train_data_loader = None
        self.test_data_loader = None

    def get_train_test_dataloader(self, batch_size, shuffle=True, split_percentage=1, num_workers=1, pin_memory=False):
        error_msg = "[!] split_percentage should be in the range [0, 1]."
        assert ((split_percentage >= 0) and (split_percentage <= 1)), error_msg

        self.__load_train_dataset()

        subset_trainA, subset_trainB = self.get_train_dataloader(batch_size=batch_size,
                                                                 split_percentage=split_percentage)

        self.train_data_loader = torch.utils.data.DataLoader(
            subset_trainA, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=pin_memory
        )

        self.__load_test_dataset()

        subset_test_dataset = ConcatDataset([subset_trainB, self.test_dataset])

        self.test_data_loader = torch.utils.data.DataLoader(
            subset_test_dataset, batch_size=batch_size, shuffle=shuffle,
            num_workers=num_workers, pin_memory=pin_memory
        )

        return self.train_data_loader, self.test_data_loader

    def get_train_dataloader(self, batch_size, split_percentage=1,
                             shuffle=True, num_workers=1, pin_memory=False):
        error_msg = "[!] split_percentage should be in the range [0, 1]."
        assert ((split_percentage >= 0) and (split_percentage <= 1)), error_msg

        self.__load_train_dataset()

        if split_percentage == 1:
            print(num_workers)
            self.train_data_loader = torch.utils.data.DataLoader(
                self.train_dataset, batch_size=batch_size, shuffle=shuffle,
                pin_memory=pin_memory)
            return self.train_data_loader
        else:
            # split_lengths = [int(len(self.train_dataset) * split_percentage),
            #                  int(len(self.train_dataset) * (1 - split_percentage))]
            #
            # subset_trainA, subset_trainB = random_split(self.train_dataset, split_lengths)

            return self.get_train_dataset_split(split_percentage=split_percentage)

    def get_train_dataset_split(self, split_percentage):
        self.__load_train_dataset()
        # split_lengths = [int(len(self.train_dataset) * split_percentage),
        #                  int(len(self.train_dataset) * (1 - split_percentage))]
        subset_trainA_length = int(len(self.train_dataset) * split_percentage)
        subset_trainB_length = len(self.train_dataset) - subset_trainA_length

        subset_trainA, subset_trainB = random_split(self.train_dataset,
                                                    [subset_trainA_length, subset_trainB_length])
        return subset_trainA, subset_trainB

    def get_test_dataset_split(self, split_percentage):
        subset_trainA, subset_trainB = self.get_train_dataset_split(split_percentage=split_percentage)
        self.__load_test_dataset()
        subset_test_dataset = ConcatDataset([subset_trainA, self.test_dataset])
        return subset_test_dataset

    def get_test_loader(self, batch_size, shuffle=False, num_workers=1, pin_memory=False):
        self.__load_test_dataset()

        self.test_data_loader = torch.utils.data.DataLoader(
            self.test_dataset, batch_size=batch_size, shuffle=shuffle,
            num_workers=num_workers, pin_memory=pin_memory
        )

        return self.test_data_loader

    def get_classes(self):
        return self.train_dataset.classes

    def __load_train_dataset(self):
        if self.dataset_name == "MNIST":
            self.__load_train_mnist_dataset(train_transform=self.__get_transform())
        elif self.dataset_name == "FashionMNIST":
            self.__load_train_fashionMnist_dataset(train_transform=self.__get_transform())

    def __load_train_mnist_dataset(self, train_transform=None):
        self.train_dataset = datasets.MNIST(
            root='./data/mnist', train=True,
            download=True, transform=train_transform,
        )

    def __load_train_fashionMnist_dataset(self, train_transform=None):
        self.train_dataset = datasets.FashionMNIST(
            root='./data/fashionMnist', train=True,
            download=True, transform=train_transform,
        )

    def __load_test_dataset(self):
        if self.dataset_name == "MNIST":
            self.__load_test_mnist_dataset(transform=self.__get_transform())
        elif self.dataset_name == "FashionMNIST":
            self.__load_test_fashionMnist_dataset(transform=self.__get_transform())

    def __load_test_mnist_dataset(self, transform=None):
        self.test_dataset = datasets.MNIST(
            root='data/mnist', train=False,
            download=True, transform=transform,
        )

    def __load_test_fashionMnist_dataset(self, transform=None):
        self.test_dataset = datasets.FashionMNIST(
            root='./data/fashionMnist', train=False,
            download=True, transform=transform,
        )

    def __get_transform(self):
        global normalize
        if self.dataset_name == "MNIST":
            normalize = self.__get_mnist_normalize_val()
        elif self.dataset_name == "FashionMNIST":
            normalize = self.__get_fashionMnist_normalize_val()

        train_transform = transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])

        return train_transform

    @staticmethod
    def __get_mnist_normalize_val():
        return transforms.Normalize((0.1307,), (0.3081,))

    @staticmethod
    def __get_fashionMnist_normalize_val():
        return transforms.Normalize((0.5,), (0.5,))

import torch
import numpy as np
from torchvision import datasets
from torchvision import transforms
from torch.utils.data.sampler import SubsetRandomSampler


class DataLoader:
    def __init__(self, dataset):
        self.dataset_name = dataset
        self.train_dataset = None
        self.validation_dataset = None
        self.test_dataset = None
        self.train_data_loader = None
        self.validation_data_loader = None
        self.test_data_loader = None

    def get_train_val_loader(self, batch_size, transform=None, augment=False,
                             validation_size=0.1, shuffle=True, num_workers=8, pin_memory=False):
        global train_transform
        error_msg = "[!] valid_size should be in the range [0, 1]."
        assert ((validation_size >= 0) and (validation_size <= 1)), error_msg

        if transform is not None:
            train_transform = transforms.Compose([
                transforms,
            ])

        if augment:
            train_transform = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                train_transform,
            ])

        if self.dataset_name == "MNIST":
            normalize = self.__get_mnist_normalize_val()
            train_transform = transforms.Compose([
                train_transform,
                transforms.Resize(32),
                transforms.ToTensor(),
                normalize,
            ])
            self.__load_train_mnist_dataset(transform=train_transform)
        elif self.dataset_name == "CIFAR10":
            normalize = self.__get_cifar10_normalize_val()
            train_transform = transforms.Compose([
                train_transform,
                transforms.ToTensor(),
                normalize,
            ])
            self.__load_train_cifar10_dataset(transform=train_transform)

        num_train = len(self.train_dataset)
        indices = list(range(num_train))
        split = int(np.floor(validation_size * num_train))

        if shuffle:
            np.random.shuffle(indices)

        train_idx, valid_idx = indices[split:], indices[:split]
        train_sampler = SubsetRandomSampler(train_idx)
        valid_sampler = SubsetRandomSampler(valid_idx)

        self.train_data_loader = torch.utils.data.DataLoader(
            self.train_dataset, batch_size=batch_size, sampler=train_sampler,
            num_workers=num_workers, pin_memory=pin_memory,
        )

        self.validation_data_loader = torch.utils.data.DataLoader(
            self.validation_dataset, batch_size=batch_size, sampler=valid_sampler,
            num_workers=num_workers, pin_memory=pin_memory,
        )

        return self.train_data_loader, self.validation_data_loader

    def get_test_loader(self, batch_size, shuffle=False, num_workers=4, pin_memory=False):
        if self.dataset_name == "MNIST":
            normalize = self.__get_mnist_normalize_val()
            test_transform = transforms.Compose([
                transforms.Resize(32),
                transforms.ToTensor(),
                normalize,
            ])
            self.__load_test_mnist_dataset(transform=test_transform)
        elif self.dataset_name == "CIFAR10":
            normalize = self.__get_cifar10_normalize_val()
            test_transform = transforms.Compose([
                transforms.ToTensor(),
                normalize,
            ])
            self.__load_test_cifar10_dataset(transform=test_transform)

        self.test_data_loader = torch.utils.data.DataLoader(
            self.test_dataset, batch_size=batch_size, shuffle=shuffle,
            num_workers=num_workers, pin_memory=pin_memory
        )

        return self.test_data_loader

    def __load_train_mnist_dataset(self, transform=None):
        self.train_dataset = datasets.MNIST(
            root='/data/mnist', train=True,
            download=True, transform=transform,
        )
        self.validation_dataset = datasets.MNIST(
            root='/data/mnist', train=True,
            download=True, transform=transform,
        )

    def __load_train_cifar10_dataset(self, transform=None):
        self.train_dataset = datasets.CIFAR10(
            root='/data/cifar10', train=True,
            download=True, transform=transform,
        )
        self.validation_dataset = datasets.CIFAR10(
            root='/data/cifar10', train=True,
            download=True, transform=transform,
        )

    def __load_test_mnist_dataset(self, transform=None):
        self.test_dataset = datasets.CIFAR10(
            root='data/mnist', train=False,
            download=True, transform=transform,
        )

    def __load_test_cifar10_dataset(self, transform=None):
        self.test_dataset = datasets.CIFAR10(
            root='data/cifar10', train=False,
            download=True, transform=transform,
        )

    @staticmethod
    def __get_mnist_normalize_val():
        return transforms.Normalize((0.1307,), (0.3081,))

    @staticmethod
    def __get_cifar10_normalize_val():
        return transforms.Normalize(
                mean=[0.4914, 0.4822, 0.4465],
                std=[0.2023, 0.1994, 0.2010],
            )
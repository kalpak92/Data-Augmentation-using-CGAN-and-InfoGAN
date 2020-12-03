import os
import torch

import convNet
from build_dataset import DataLoader
from train import TrainCnn
from test import TestCnn
from utils import configParams, get_device, save_checkpoint, load_checkpoint, save_dict_to_json


class ConvNetManger():
    def __init__(self, dataset_name, model_dir='experiments/base_cnn'):
        self.dataset_name = dataset_name
        self.dataloader = DataLoader(dataset_name)
        self.model_dir = model_dir
        self.config_params = None

    def get_train_val_dataLoader(self, num_workers=1, pin_memory=False):
        train_dl = self.dataloader.get_train_val_dataloader(batch_size=self.config_params.batch_size,
                                                            num_workers=num_workers, pin_memory=pin_memory,
                                                            validation_size=0.1)
        return train_dl

    def get_test_dataLoader(self, num_workers=1, pin_memory=False):
        test_dl = self.dataloader.get_test_loader(batch_size=self.config_params.batch_size,
                                                  num_workers=num_workers, pin_memory=pin_memory)
        return test_dl

    def initialize_weights_biases(self, model):
        if isinstance(model, torch.nn.Conv2d):
            torch.nn.init.kaiming_normal_(model.weight.data, nonlinearity='relu')
            torch.nn.init.constant_(model.bias.data, 0)
        elif isinstance(model, torch.nn.Linear):
            torch.nn.init.xavier_normal_(model.weight.data, gain=torch.nn.init.calculate_gain('relu'))
            torch.nn.init.constant_(model.bias.data, 0)

    def get_model_config(self):
        json_path = os.path.join(self.model_dir, 'config.json')
        assert os.path.isfile(
            json_path), "No json configuration file found at {}".format(json_path)
        self.config_params = configParams(json_path)
        self.config_params.cuda = torch.cuda.is_available()
        self.config_params.device = get_device()
        self.config_params.loss_plot_path = os.path.join(self.model_dir, 'loss_plot.jpeg')

        if self.dataset_name == "MNIST":
            self.config_params.num_channels = 1
        elif self.dataset_name == "CIFAR10":
            self.config_params.num_channels = 3

    def load_train_data(self):
        print("Device: ", self.config_params.device)
        if torch.cuda.is_available():
            self.config_params.train_dataloader = self.get_train_val_dataLoader(num_workers=4, pin_memory=True)
        else:
            self.config_params.train_dataloader = self.get_train_val_dataLoader()

    def load_test_data(self):
        print("Device: ", self.config_params.device)
        if torch.cuda.is_available():
            self.config_params.test_dataloader = self.get_test_dataLoader(num_workers=4, pin_memory=True)
        else:
            self.config_params.test_dataloader = self.get_test_dataLoader()

    def convNet_train(self):
        model = convNet.ConvNet(self.config_params).to(self.config_params.device)
        model.apply(self.initialize_weights_biases)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.config_params.learning_rate)
        print("Starting training for {} epoch(s)".format(self.config_params.num_epochs))

        convNet_train = TrainCnn()
        convNet_train.train(model, optimizer, self.config_params)

        save_checkpoint({'state_dict': model.state_dict(),
                         'optimizer_dict': optimizer.state_dict()},
                        checkpoint=self.model_dir)

    def convNet_test(self):
        model = convNet.ConvNet(self.config_params).to(self.config_params.device)
        load_checkpoint(checkpoint=os.path.join(self.model_dir + '/last.pth.tar'), model=model, optimizer=False)
        metrics = convNet.metrics

        convNet_test = TestCnn()
        test_metrics = convNet_test.test(model, metrics, self.config_params)

        save_path = os.path.join(self.model_dir, "metrics_test.json")
        save_dict_to_json(test_metrics, save_path)


if __name__ == '__main__':
    network_manager = ConvNetManger("MNIST")
    network_manager.get_model_config()
    network_manager.load_train_data()
    #network_manager.convNet_train()

    network_manager.load_test_data()
    network_manager.convNet_test()

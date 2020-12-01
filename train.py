import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F


class TrainCnn():
    def __init__(self, model, optimizer=None, scheduler=None, load_path=None):
        self.metrics = []
        self.model = model
        if load_path is not None:
            self.model = torch.load(load_path)
        self.optimizer = optimizer
        self.scheduler = scheduler

    def initialize_weights_biases(m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight.data, nonlinearity='relu')
            nn.init.constant_(m.bias.data, 0)
        elif isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight.data, gain=nn.init.calculate_gain('relu'))
            nn.init.constant_(m.bias.data, 0)

    def run(self, scheduler, epochs, train_loader):
        print("Start Training...")

        for epoch in range(epochs):
            if scheduler is not None:
                scheduler.step()

            epoch_loss = 0
            correct = 0

            for batch_idx, (data, label) in enumerate(train_loader):
                self.optimizer.zero_grad()

                X = Variable(data.view(-1, 784))
                Y = Variable(label)

                out = self.model(X)

                pred = out.data.max(1, keepdim=True)[1]
                predicted = pred.eq(Y.data.view_as(pred))

                correct += predicted.sum()
                loss = F.nll_loss(out, Y)
                loss.backward()
                self.optimizer.step()
                epoch_loss += loss.item()

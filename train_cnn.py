import torch
from torch.autograd import Variable
import torch.nn.functional as F

class TrainCnn():
    def __init__(self, model, optimizer=None, scheduler=None, load_path=None):
        self.metrics = []
        self.model = model
        if load_path is not None:
            self.model = torch.load(load_path)
        self.optimizer = optimizer
        self.scheduler = scheduler

    def save_model(self, path):
        torch.save(self.model.state_dict(), path)

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

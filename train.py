from torch.autograd import Variable
from tqdm import tqdm
from convNet import loss_fn
from utils import RunningAverageLoss, plot_loss_epoch


class TrainCnn():
    def train(self, model, optimizer, params):
        train_data_loader = params.train_dataloader
        loss_train = []
        print("Start Training CNN ...")
        model.train()

        for epoch in range(params.num_epochs):
            # Run one epoch
            print("Epoch {}/{}".format(epoch + 1, params.num_epochs))
            loss_avg = RunningAverageLoss()
            loss_total = 0
            # Use tqdm for progress bar
            with tqdm(total=len(train_data_loader)) as t:
                for i, (train_batch, label_batch) in enumerate(train_data_loader):
                    # move to GPU if available
                    if params.cuda:
                        train_batch, label_batch = train_batch.cuda(
                            non_blocking=True), label_batch.cuda(non_blocking=True)

                    # convert to torch Variables
                    train_batch, label_batch = Variable(
                        train_batch), Variable(label_batch)

                    # compute model output and loss
                    output_batch = model(train_batch)
                    loss = loss_fn(output_batch, label_batch)

                    # clear previous gradients, compute gradients of all variables wrt loss
                    optimizer.zero_grad()
                    loss.backward()

                    # performs updates using calculated gradients
                    optimizer.step()

                    # update the average loss and total loss
                    loss_avg.update(loss.item())
                    loss_total += loss.item()

                    t.set_postfix(loss='{:05.3f}'.format(loss_avg()))
                    t.update()

                loss_train.append(loss_total)

        plot_loss_epoch(loss_train, params.loss_plot_path)
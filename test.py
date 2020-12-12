import numpy as np
import torch
from torch.autograd import Variable
from convNet import loss_fn


class TestCnn():
    def test(self, model, metrics, params):
        test_data_loader = params.test_dataloader

        # Set model to evaluation mode
        model.eval()

        # summary of current evaluation loop
        summary = []
        predictions = torch.tensor([])
        labels = torch.tensor([])

        for data_batch, labels_batch in test_data_loader:
            # move to GPU if available
            if params.cuda:
                data_batch, labels_batch = data_batch.cuda(non_blocking=True), \
                                           labels_batch.cuda(non_blocking=True)
            # fetch the next evaluation batch
            data_batch, labels_batch = Variable(data_batch), Variable(labels_batch)

            # compute model output
            output_batch = model(data_batch)
            loss = loss_fn(output_batch, labels_batch)

            _, y_pred_tags = torch.max(output_batch, dim=1)
            predictions = torch.cat((predictions, y_pred_tags.cpu()), dim=0)
            labels = torch.cat((labels, labels_batch.cpu()), dim=0)

            # extract data from torch Variable, move to cpu, convert to numpy arrays
            # output_batch = output_batch.data.cpu().numpy()
            # labels_batch = labels_batch.data.cpu().numpy()

            # compute all metrics on this batch
            # summary_batch = {metric: metrics[metric](output_batch, labels_batch)
            #                  for metric in metrics}
            # summary_batch['loss'] = loss.item()
            # summary.append(summary_batch)

        # compute mean of all metrics in summary
        # metrics_mean = {metric: np.mean([x[metric]
        #                                  for x in summary]) for metric in summary[0]}

        return predictions.numpy(), labels.numpy()

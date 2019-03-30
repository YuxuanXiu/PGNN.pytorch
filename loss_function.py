__author__ = 'YuxuanXiu'

import torch
import numpy as np


def empirical_error(Y_, Y):
    criterion = torch.nn.MSELoss(reduction='mean')
    output = criterion(Y_, Y)
    return output


def structural_error(model, lambda_1, lambda_2):
    output = torch.tensor(0., requires_grad=True)
    for name, p in model.named_parameters():
        if 'weight' in name:
            output = output + lambda_1 * torch.sum(torch.abs(p)) + lambda_2 * torch.norm(p)
    return output


def physical_inconsistency_batch(batch_data):
    # The input batch_data is a batch_size*14 list, the first 12 columns are input variables
    # The 13th column is the label and the 14th column is the prediction
    index = np.lexsort([batch_data[:, 1]])
    batch_data = batch_data[index, :]
    for i, each_line in enumerate(batch_data):
        Y_ = each_line[-1]
        density = 1000 * (1 - (Y_ + 288.9414) * (Y_ - 3.9863) ** 2 / (508929.2 * (Y_ + 68.12963)))
        batch_data[i].append(density)
    # Second column(Depth), ascending order
    for i in range(0, batch_data.shape[0]):
        delta = batch_data[i+1, -1] - batch_data[i, -1]






    return output

def physics_based_loss_funtion():
    return output
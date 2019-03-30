__author__ = 'YuxuanXiu'

import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader, TensorDataset
from experimental_design import train_batch_size,test_batch_size

# Suppose that data is a 20615*13 matrix.
# Each row is a temperature observation, 7072 for lake Mille Lacs and 13543 for Mendota.
# Column names are listed as follows: Day of Year, Depth, Short-wave Radiation, Long-wave Radiation,
# Air Temperature, Relative Humidity, Wind Speed, Rain, Growing Degree Days, Is Freezing, Is Snowing,
# The Result of General Lake Model, Ground-truth.

training_set = "train_set.csv.gz"
validating_set = "validate_set.csv.gz"
testing_set = "test_set.csv.gz"


def generate_data_loader(dataset,batch_size):
    xy = np.loadtxt(dataset, delimiter=',', dtype=np.float32)
    x_data = torch.from_numpy(xy[:, 0:-1])
    y_data = torch.from_numpy(xy[:, [-1]])
    dataset = TensorDataset(x_data, y_data)
    # !!!ATTENTION!!!
    # In order to use physical inconsistency loss correctly, shuffle must be set false,
    # and data from two lakes should not be mixed!
    # The data should be sorted by time-step with ascending order.
    data_loader = DataLoader(dataset, batch_size, shuffle=False, num_workers=4)
    return data_loader


train_data_loader = generate_data_loader(training_set, train_batch_size)
val_data_loader = generate_data_loader(validating_set, train_batch_size)
test_data_loader = generate_data_loader(testing_set, test_batch_size)




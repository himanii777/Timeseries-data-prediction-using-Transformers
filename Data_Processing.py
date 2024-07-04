import pandas as pd
import numpy as np
import torch
import torch
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn
from matplotlib import pyplot as plt
import math
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import mean_squared_error

data= pd.read_csv('trafficdata.csv', header=None) #change this to your situation

#We gotta slice the data as in a typical time series prediction task

def load_data(data, input_seq_length=256, output_seq_length=256, offset=0):
    #input_seq_length -> #data we process to predict (1,....k-1)
    #output_seq_length -> #data we predict after we process (k,...,n)
    #offset -> gap between k-1 and the first data of the output_seq_length
    #in my case I set offset to 0

    N = data.shape[1]  # Total number of sequence
    k = N - (input_seq_length + output_seq_length+offset) # Maximum index for slicing

    in_slices = np.array([data.iloc[:, i:(i + input_seq_length)].values for i in range(k + 1)])
    out_slices = np.array([data.iloc[:, (i + input_seq_length+ offset):(i + input_seq_length + output_seq_length+  offset)].values for i in range(k + 1)])
    train_size = int((in_slices.shape[1]) * 0.8)
    train_in_data, test_in_data = in_slices[:, :train_size], in_slices[:, train_size:]
    train_out_data, test_out_data = out_slices[:, :train_size], out_slices[:, train_size:]

    train_in_data = torch.tensor(train_in_data, dtype=torch.float32)
    test_in_data= torch.tensor(test_in_data, dtype=torch.float32)
    train_out_data = torch.tensor(train_out_data, dtype=torch.float32)
    test_out_data = torch.tensor(test_out_data, dtype=torch.float32)

    train_dataset = TensorDataset(train_in_data, train_out_data)
    test_dataset= TensorDataset(test_in_data, test_out_data )

    train_dataloader = DataLoader(train_dataset, batch_size=100, shuffle=False)
    test_dataloader = DataLoader(test_dataset, batch_size=100, shuffle=False)

    return train_dataloader, test_dataloader



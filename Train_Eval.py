
import numpy as np
import torch
import torch
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn
from matplotlib import pyplot as plt
import math
from sklearn.metrics import mean_squared_error
from Model import TransformerModel
from Data_Processing import load_data



model = TransformerModel().to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
train_dataloader, test_dataloader= load_data(data)

def train(model, train_data_loader, criterion, optimizer, epochs):
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for train_in_data, train_out_data in train_data_loader:
            train_in_data, train_out_data = train_in_data.to(device), train_out_data.to(device)
            outputs = model(train_in_data)
            loss = criterion(outputs, train_out_data)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f'Epoch [{epoch+1}/{epochs}], Loss: {total_loss/len(train_data_loader):.4f}')

train(model, train_dataloader, criterion, optimizer, epochs=100)



def evaluate_model(model, test_data_loader):
    model.eval()
    model.to('cpu')
    predictions, actuals = [], []

    with torch.no_grad():
        for test_in_data, test_out_data in test_data_loader:
            outputs = model(test_in_data.to('cpu'))
            predictions.append(outputs.numpy())
            actuals.append(test_out_data.numpy())

    predictions = np.concatenate(predictions, axis=0).flatten()
    actuals = np.concatenate(actuals, axis=0).flatten()
    mse = mean_squared_error(actuals, predictions)
    return mse
error=evaluate_model(model, test_dataloader)
print(f"Mean Squared Error is : {error}")
import torch
import matplotlib.pyplot as plt
from Model import TransformerModel
from Data_Processing import load_data


# train_dataloader, test_dataloader= load_data(data)
# model = TransformerModel().to(device)

def predict(model, test_data_loader, batch_size):
    model.eval()
    model.to('cpu')
    for i, (test_in_data, test_out_data) in enumerate(test_data_loader):
        if i == batch_size:
            break
    input = test_in_data[0, :, :].unsqueeze(0)
    target = test_out_data[0, :, :].unsqueeze(0)

    with torch.no_grad():
        prediction = model(input)
    prediction_np = prediction.flatten().numpy()
    target_np = target.flatten().numpy()

    return prediction_np, target_np

predictions, targets = predict(model, test_dataloader, 10)


def plot_results(predictions, targets):
    plt.figure(figsize=(14, 7))
    plt.plot(targets, label='Actual Data', color='blue')
    plt.plot(predictions, label='Predicted Data', color='red', linestyle='--')
    plt.title('Comparison of Actual and Predicted Data')
    plt.xlabel('Time Step')
    plt.ylabel('Traffic_data')
    plt.legend()
    plt.grid(True)
    plt.show()
plot_results( predictions[100:200], targets[100:200])
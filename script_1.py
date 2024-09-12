import os
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import tensorflow as tf
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

#tensorboard --logdir=runs

# Load MNIST dataset using torchvision
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST('./data', train=False, transform=transform)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1000, shuffle=False)

# Define the model with a handrolled attention mechanism
class AttentionModel(nn.Module):
    def __init__(self):
        super(AttentionModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(7 * 7 * 64, 128)
        self.attention = nn.Linear(128, 128)  # Attention mechanism
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = nn.functional.relu(self.conv1(x))
        x = nn.functional.max_pool2d(x, 2)
        x = nn.functional.relu(self.conv2(x))
        x = nn.functional.max_pool2d(x, 2)
        x = x.view(-1, 7 * 7 * 64)
        x = nn.functional.relu(self.fc1(x))

        # Apply attention
        attention_weights = nn.functional.softmax(self.attention(x), dim=1)
        x = x * attention_weights

        x = self.fc2(x)
        return x

model = AttentionModel()

# Define a custom loss function (example: weighted cross-entropy)
class WeightedCrossEntropyLoss(nn.Module):
    def __init__(self, weights):
        super(WeightedCrossEntropyLoss, self).__init__()
        self.weights = weights

    def forward(self, inputs, targets):
        log_probs = nn.functional.log_softmax(inputs, dim=1)
        loss = -torch.sum(self.weights[targets] * log_probs[range(len(targets)), targets]) / len(targets)
        return loss

# Example weights (adjust as needed)
weights = torch.ones(10)
criterion = WeightedCrossEntropyLoss(weights)
optimizer = optim.Adam(model.parameters(), lr=0.001)
writer = SummaryWriter()  # Create a SummaryWriter instance

# Training loop with tqdm and TensorBoard logging
for epoch in range(2):
    with tqdm(train_loader, unit="batch") as tepoch:
        tepoch.set_description(f"Epoch {epoch+1}")
        for batch_idx, (data, target) in enumerate(tepoch):
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            writer.add_scalar('Training Loss', loss.item(), epoch * len(train_loader) + batch_idx)
            tepoch.set_postfix(loss=loss.item())

# Testing loop
correct = 0
total = 0
with torch.no_grad():
    for data, target in test_loader:
        output = model(data)
        _, predicted = torch.max(output.data, 1)
        total += target.size(0)
        correct += (predicted == target).sum().item()

    # Log test accuracy to TensorBoard after each epoch
    accuracy = 100 * correct / total
    writer.add_scalar('Accuracy/test', accuracy, epoch)

print('Accuracy on test set: %d %%' % (100 * correct / total))
writer.close() 

#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# The neural network that is used
class fraudDetectionNN(nn.Module):
    def __init__(self, input_dim):
        super(fraudDetectionNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.sigmoid(self.fc4(x))
        return x


def neuralNetworkModel(data):

    # prepare training data
    X = data.drop(columns=["Class", "timeHour24", "Time", "timeHour48"]).values
    y = data["Class"].values


    # split data into train, eval and test
    X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
    )
    X_train, X_val, y_train, y_val = train_test_split(
    X_train, y_train, test_size=0.25, random_state=42
    )

    # convert the data into the right format
    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)  # shape (n,1)
    X_val = torch.tensor(X_val, dtype=torch.float32)
    y_val = torch.tensor(y_val, dtype=torch.float32).unsqueeze(1)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)

    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=512, shuffle=True)

    # initialize model, loss, optimizer
    model = fraudDetectionNN(input_dim=X.shape[1])
    criterion = nn.BCELoss()  # binary cross-entropy
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    # training loop
    epochs = 100
    patience = 10   # early stop if no improvement after 10 epochs
    best_loss = float("inf")
    counter = 0
    for epoch in range(epochs):
        # Training
        model.train()
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

        # validation
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val)
            val_loss = criterion(val_outputs, y_val).item()

        print(f"Epoch [{epoch+1}/{epochs}] - Validation Loss: {val_loss:.4f}")

        # check for improvement
        if val_loss < best_loss:
            best_loss = val_loss
            counter = 0
            best_model_state = model.state_dict()  # save best model
        else:
            counter += 1
            if counter >= patience:
                print("Early stopping triggered!")
                model.load_state_dict(best_model_state)  # restore best model
                break

    # evaluate
    model.eval()
    with torch.no_grad():
        y_pred = model(X_test)
        y_pred = (y_pred >= 0.1).int()

    # convert to numpy
    y_true = y_test.numpy()
    y_pred = y_pred.numpy()


    print("Neural Network Results:")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))











# BIG Neural Network Results:
# Accuracy: 0.9993504441557529
# Confusion Matrix:
#  [[56848    16]
#  [   21    77]]
#               precision    recall  f1-score   support
#
#          0.0       1.00      1.00      1.00     56864
#          1.0       0.83      0.79      0.81        98
#
#     accuracy                           1.00     56962
#    macro avg       0.91      0.89      0.90     56962
# weighted avg       1.00      1.00      1.00     56962








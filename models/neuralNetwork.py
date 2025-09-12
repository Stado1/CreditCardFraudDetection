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


class SimpleNN(nn.Module):
    def __init__(self, input_dim):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 2048)
        self.fc2 = nn.Linear(2048, 1024)
        self.fc3 = nn.Linear(1024, 512)
        self.fc4 = nn.Linear(512, 256)
        self.fc5 = nn.Linear(256, 128)
        self.fc6 = nn.Linear(128, 64)
        self.fc7 = nn.Linear(64, 32)
        self.fc8 = nn.Linear(32, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(0.4)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.relu(self.fc3(x))
        x = self.dropout(x)
        x = self.relu(self.fc4(x))
        x = self.dropout(x)
        x = self.relu(self.fc5(x))
        x = self.dropout(x)
        x = self.relu(self.fc6(x))
        x = self.dropout(x)
        x = self.relu(self.fc7(x))
        x = self.dropout(x)
        x = self.sigmoid(self.fc8(x))
        return x


# class SimpleNN(nn.Module):
#     def __init__(self, input_dim):
#         super(SimpleNN, self).__init__()
#         self.fc1 = nn.Linear(input_dim, 32)
#         self.fc2 = nn.Linear(32, 16)
#         self.fc3 = nn.Linear(16, 1)
#         self.relu = nn.ReLU()
#         self.sigmoid = nn.Sigmoid()
#
#     def forward(self, x):
#         x = self.relu(self.fc1(x))
#         x = self.relu(self.fc2(x))
#         x = self.sigmoid(self.fc3(x))
#         return x


def neuralNetworkModel(data):

    X = data.drop(columns=["Class", "timeHour24", "Time", "timeHour48"]).values
    # X = data.drop(columns=["Class", "Time"]).values
    y = data["Class"].values


    X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
    )
    X_train, X_val, y_train, y_val = train_test_split(
    X_train, y_train, test_size=0.25, random_state=42
    )


    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)  # shape (n,1)
    X_val = torch.tensor(X_val, dtype=torch.float32)
    y_val = torch.tensor(y_val, dtype=torch.float32).unsqueeze(1)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)

    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=512, shuffle=True)



    model = SimpleNN(input_dim=X.shape[1])

    # --- Loss & Optimizer ---
    criterion = nn.BCELoss()  # binary cross-entropy
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    # --- Training Loop ---
    epochs = 100
    patience = 10   # stop if no improvement after 5 epochs
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

        # Validation
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val)
            val_loss = criterion(val_outputs, y_val).item()

        print(f"Epoch [{epoch+1}/{epochs}] - Validation Loss: {val_loss:.4f}")

        # Check for improvement
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

    # --- Evaluation ---
    model.eval()
    with torch.no_grad():
        y_pred = model(X_test)
        y_pred = (y_pred >= 0.5).int()

    # Convert back to numpy for sklearn metrics
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








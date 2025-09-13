#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler

def logisticRegressionModel(data):

    # prepare training data
    X = data.drop(columns=["Class", "timeHour24", "Time", "timeHour48"]).values
    y = data["Class"].values

    # scale data to speed up training
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # split data into train and test
    X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
    )

    # create and train model
    model = LogisticRegression(max_iter=500)
    model.fit(X_train, y_train)

    # predict test data
    y_pred = model.predict(X_test)

    # evaluate
    print("Logistic Regresion Results:")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("Classification Report:\n", classification_report(y_test, y_pred))



















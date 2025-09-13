#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import IsolationForest
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

def isolationForrestModel(data):

    # prepare training data
    X = data.drop(columns=["Class", "timeHour24", "Time", "timeHour48"]).values
    y = data["Class"].values

    # split data into train and test
    X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
    )

    # create and train model
    iso = IsolationForest(
        n_estimators=10000,
        contamination=0.1,    # estimate fraction of anomalies in your dataset
        random_state=42
    )
    iso.fit(X_train)

    # predict test data
    y_pred = iso.predict(X_test)
    # transofrm predicions for evaluation
    y_pred = [0 if val == 1 else 1 for val in y_pred]

    # evaluate
    print("Isolation Forrest Results:")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))



















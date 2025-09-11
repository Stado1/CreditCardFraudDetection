#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import IsolationForest
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

def isolationForrestModel(data):

    X = data.drop(columns=["Class", "timeHour24", "Time", "timeHour48"]).values
    # X = data.drop(columns=["Class", "Time"]).values
    y = data["Class"].values

    X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
    )

    iso = IsolationForest(
        n_estimators=200,
        contamination=0.00173,    # estimate fraction of anomalies in your dataset
        random_state=42
    )
    iso.fit(X_train)

    # --- Predictions ---
    y_pred = iso.predict(X_test)

    y_pred = [0 if val == 1 else 1 for val in y_pred]

    print("Isolation Forrest Results:")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))



















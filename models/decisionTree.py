#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

def decisionTreeModel(data):

    X = data.drop(columns=["Class", "timeHour24", "Time", "timeHour48"]).values
    # X = data.drop(columns=["Class", "Time"]).values
    y = data["Class"].values

    X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
    )

    model = DecisionTreeClassifier(random_state=42, max_depth=8)
    model.fit(X_train, y_train)
    y_pred= model.predict(X_test)

    print("Decision Tree Results:")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))



















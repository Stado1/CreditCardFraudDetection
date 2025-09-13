#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

def randomForrestModel(data):

    # prepare training data
    X = data.drop(columns=["Class", "timeHour24", "Time", "timeHour48"]).values
    y = data["Class"].values

    # split data into train and test
    X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
    )

    # create and train model
    numEstimators = 60
    rf = RandomForestClassifier(
        n_estimators=numEstimators,
        max_depth=None,
        random_state=42
    )
    rf.fit(X_train, y_train)

    # predict test data
    y_pred = rf.predict(X_test)

    # evaluate
    print("Random Forrest Results:")
    print("numEstimators = ", numEstimators)
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))



















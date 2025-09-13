#!/usr/bin/env python3

import pandas as pd
import sklearn
import torch
import numpy as np
import math

from dataAnalysis import graphCorrelationAmountAndFraud
from dataAnalysis import graphCorrelationTimeAndFraud
from dataAnalysis import graphCorrelationTimeAndTransaction
from dataAnalysis import splitDataBasedOnFraud
from dataAnalysis import graphVxValues

from models import logisticRegressionModel
from models import naiveBayesModel
from models import decisionTreeModel
from models import randomForrestModel
from models import isolationForrestModel
from models import neuralNetworkModel

data = pd.read_csv('creditcard.csv')

print("general info of data")
print("amount of transactions: ")
print(data.shape[0])
fraud_count = data[data['Class'] == 1].shape[0]
print("amount of frauds: ")
print(fraud_count)
print("frauds/transactions = ", fraud_count/data.shape[0])
print("amount of data: ")
print(data.shape[1])

# convert time to 24 hour and 48 hour cycle and add as columns
data['timeHour24'] = (data['Time'] / 3600).round().astype(int) % 24
data['timeHour48'] = (data['Time'] / 3600).round().astype(int)



print("----------------------")
print("creating graphs")
graphCorrelationAmountAndFraud(data)
graphCorrelationTimeAndFraud(data)
graphCorrelationTimeAndTransaction(data)

fraudData, noFraudData = splitDataBasedOnFraud(data)
graphVxValues(fraudData, noFraudData, 14)
print("graphs saved")

# print("----------------------")
# print("training logistic regression model")
# logisticRegressionModel(data)

# print("----------------------")
# print("training naive bayes model")
# naiveBayesModel(data)

# print("----------------------")
# print("training decision tree")
# decisionTreeModel(data)

# print("----------------------")
# print("training random forrest")
# randomForrestModel(data)

# print("----------------------")
# print("training isolation forrest")
# isolationForrestModel(data)

# print("----------------------")
# print("training neural network")
# neuralNetworkModel(data)






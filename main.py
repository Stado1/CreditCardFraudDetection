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

# print general info of data
# print("amount of transactions: ")
# print(data.shape[0])
# fraud_count = data[data['Class'] == 1].shape[0]
# print("amount of frauds: ")
# print(fraud_count)
# print("frauds/transactions = ", fraud_count/data.shape[0])
# print("amount of data: ")
# print(data.shape[1])

# convert time to 24 hour cycle and add as column
data['timeHour24'] = (data['Time'] / 3600).round().astype(int) % 24
data['timeHour48'] = (data['Time'] / 3600).round().astype(int)

# print(data.sort_values(by="Time", ascending=False))



print("----------------------")

#
# graphCorrelationAmountAndFraud(data)
# graphCorrelationTimeAndFraud(data)
# graphCorrelationTimeAndTransaction(data)
#
# fraudData, noFraudData = splitDataBasedOnFraud(data)
#
# graphVxValues(fraudData, noFraudData, 14)


# logisticRegressionModel(data)
# print("----------------------")
# naiveBayesModel(data)
# print("----------------------")
# decisionTreeModel(data)
# print("----------------------")
# randomForrestModel(data)
# print("----------------------")
# isolationForrestModel(data)
# print("----------------------")
neuralNetworkModel(data)






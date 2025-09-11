#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt


# this function will plot the amount of frauds vs the transaction amount
# it will plot the transaction amounts in bins of 100 until 10000
def graphCorrelationAmountAndFraud(data):

    dataToPlot = []

    # loop trough each bin and find the amount of frauds that occured for that transaction amount
    for i in range(100):
        lower = 100 * i
        upper = 100 * (i + 1)
        filterData = data[(data['Amount'] >= lower)]
        filterData = filterData[(filterData['Amount'] < upper)]

        fraudCount = filterData[filterData['Class'] == 1].shape[0]
        dataToPlot.append(fraudCount)


    # make a plot and save it
    arr = np.arange(0, 10000, 100)
    plt.figure()
    plt.plot(arr, dataToPlot, marker='o' )
    plt.title("Frauds vs Amount")
    plt.xlabel("Amount")
    plt.ylabel("Frauds")
    plt.grid(True)
    plt.savefig("graphFraudVsAmount.png")


# this function will plot the amount of frauds vs the time they occured
# it will plot the amount of frauds that happen per hour
def graphCorrelationTimeAndFraud(data):

    dataToPlot = []

    # loop trough each hour and find how many frauds happend at that time
    for i in range(24):
        filterData = data[(data['timeHour'] == i)]
        fraudCount = filterData[filterData['Class'] == 1].shape[0]
        dataToPlot.append(fraudCount)

    # make a plot and save it
    arr = np.arange(0, 24)
    plt.figure()
    plt.plot(arr, dataToPlot, marker='o' )
    plt.title("Frauds vs Time")
    plt.xlabel("Time (hour)")
    plt.ylabel("Frauds")
    plt.grid(True)
    plt.savefig("graphFraudVsTime.png")




# this function will plot the amount of transactions vs the time they occured
# it will plot the amount of transactions that happen per hour
def graphCorrelationTimeAndTransaction(data):

    dataToPlot = []

    # loop trough each hour and find how many transactions happend at that time
    for i in range(24):
        filterData = data[(data['timeHour'] == i)]
        transCount = filterData.shape[0]
        dataToPlot.append(transCount)

    # make a plot and save it
    arr = np.arange(0, 24)
    plt.figure()
    plt.plot(arr, dataToPlot, marker='o' )
    plt.title("transactions vs Time")
    plt.xlabel("Time (hour)")
    plt.ylabel("Transactions")
    plt.grid(True)
    plt.savefig("graphTransactionsVsTime.png")























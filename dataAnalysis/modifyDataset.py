#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt


# this function will split the data set up into 2 datasets
# one dataset with all the frauds and one dataset with no frauds
def splitDataBasedOnFraud(data):
    fraudData = data[data['Class'] == 1]
    noFraudData = data[data['Class'] == 0]

    return fraudData, noFraudData












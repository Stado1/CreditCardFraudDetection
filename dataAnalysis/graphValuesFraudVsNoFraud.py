#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt

#1 =< x =< 28
def graphVxValues(fraudData, noFraudData, x):
    # Plot histograms for both
    col = f"V{x}"
    plt.hist(noFraudData[col], bins=50, alpha=0.6, label="noFraud", density=True)
    plt.hist(fraudData[col], bins=50, alpha=0.6, label="Fraud", density=True)

    plt.title(f"Distribution of {col} (Fraud vs noFraud)")
    plt.xlabel(col)
    plt.ylabel("Density")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.savefig("graphVvalueFraudVsNoFraud.png")

























# CreditCardFraudDetection
This project tries different methods to find credit card fraud on the Kaggle dataset. The goal for each method is to get an as high as possible F1 score for both fraud and no fraud classes and to get an as low as possible ratio between false positives / true negatives (positive = no fraud, negative = fraud). Both measurements are used because for a fraud detection system false poistives are worse than false negatives.

### Necessary packages to run this code:
- scikit-learn             1.7.1
- torch                    2.8.0
- pandas                   2.3.2

### How to run the code:
Run the "Main.py" file.


## Workflow
### Data Preprocessing
Before training the model some data exploration and preprocessing is done:
- The time data is converted in the dataset. The time is converted from seconds to hours, both the total 48 hours and in a 24 hour cycle.
- A graph of the transactions amount vs the amount of frauds is plotted
- Graphs of the time vs the amount of frauds is plotted.
- Graphs of the time vs the amount of transactions is plotted.
- Histograms of each PCA value is plotted for the fraud and no fraud data, in the same graph.

### Models
The following models will be used:
- Logistic regression
- Naïve Bayes
- Decision tree
- Random forrest
- Isolation forrest
- Fully connected neural network

## Results
### Usable data
- For each PCA value, there is a clear difference between the fraud and no fraud case. This means that this data can be used to train the models.
- For the transaction amount it can also be seen that frauds happened more for lower transaction amounts. So this data can also be used.
- For the time data, the original in seconds, the 24 hours and the 48 hours, there did not seem to be a pattern. So this data will not be used.

For all models there will be a train test split of 0.8/0.2.

### Logistic Regression
Logistic regression is a simple model, only one tunable parameter is used. This parameter is the maximum amount of iterations, which is set to 1000.

No fraud F1: 1.00. Fraud F1: 0.70. The ratio FP/TN = 41/57 = 0.72.

### Naïve Bayes
Naive Bayes is also a simple model, no tunable parameters are used.

No fraud F1: 0.99, Fraud F1: 0.11, The ratio FP/TN = 18/80 = 0.23.

### Decision Tree
The decision tree model will only use one tunable parameter which is max depth. Multiple values were tried and the results can be seen in the table.

| Max depth | No fraud F1 | Fraud F1 | FP/TN        |
|-----------|-------------|----------|--------------|
| 5         | 1.00        | **0.84**     | 21/77 = 0.27 |
| 7         | 1.00        | **0.84**     | **20/78 = 0.26** |
| 9         | 1.00        | **0.84**     | **20/78 = 0.26** |
| 11        | 1.00        | 0.79     | 24/74 = 0.32 |
| 13        | 1.00        | 0.77     | 23/75 = 0.31 |
| 15        | 1.00        | 0.77     | 22/76 = 0.29 |
| 20        | 1.00        | 0.75     | 22/76 = 0.29 |
| 30        | 1.00        | 0.75     | 22/76 = 0.29 |

### Random Forrest
The random forrest model will use 2 tunable parameters: number of estimators and max depth. Max depth will be set to None because in a forrest it is not likely to cause overfitting. Different value for number of estimators will be used and the results can be seen in the table. 

| Num estimators | No fraud F1 | Fraud F1 | FP/TN        |
|----------------|-------------|----------|--------------|
| 5              | 1.00        | 0.84     | 24/74 = 0.32 |
| 7              | 1.00        | 0.84     | 24/74 = 0.32 |
| 10             | 1.00        | 0.84     | 25/73 = 0.34 |
| 15             | 1.00        | **0.86** | **22/76 = 0.29** |
| 30             | 1.00        | **0.86** | 23/75 = 0.31 |
| 60             | 1.00        | **0.86** | 23/75 = 0.31 |

### Isolation Forrest
The isolation forrest will also use 2 tunable parameters: number of estimators and contamination. When the contamination is not used the results can be seen in the table.

| Num estimators | No fraud F1 | Fraud F1 | FP/TN            |
|----------------|-------------|----------|------------------|
| 100            | 0.98        | 0.07     | 20/78 = 0.26     |
| 200            | 0.98        | 0.07     | 18/80 = 0.23     |
| 500            | 0.98        | 0.08     | 18/80 = 0.23     |
| 1000           | 0.98        | 0.07     | 17/81 = 0.21     |
| 10000          | 0.98        | 0.07     | 17/81 = 0.21     |

Contamination is the estimated amount of fraud cases as a fraction of the total transaction. If this is set equal to the actual fraction: 0.00173, the results can be seen in the table.

| Num estimators | No fraud F1 | Fraud F1 | FP/TN            |
|----------------|-------------|----------|------------------|
| 100            | **1.00**        | 0.25     | 73/25 = 2.92     |
| 200            | **1.00**        | 0.27     | 71/27 = 2.63     |
| 500            | **1.00**        | **0.30**     | 69/29 = 2.40     |
| 1000           | **1.00**        | **0.30**     | 69/29 = 2.40     |
| 10000          | **1.00**        | 0.29     | 70/28 = 2.50     |

By over estimating the contamination to 0.1 the amount of false positives can be reduced at the cost of an increase in the amount of false negatives. The results can be seen in the table.


| Num estimators | No fraud F1 | Fraud F1 | FP/TN            |
|----------------|-------------|----------|------------------|
| 100            | 0.95        | 0.03     | 9/89 = 0.10     |
| 200            | 0.95        | 0.03     | 8/90 = 0.09     |
| 500            | 0.95        | 0.03     | 8/90 = 0.09     |
| 1000           | 0.95        | 0.03     | **7/91 = 0.08**     |
| 10000          | 0.95        | 0.03     | 8/90 = 0.09     |


### Neural Network
The neural netqwork has a lot of tunable parameters, the ones that will be explored are the amount of neurons, the amount of layers and the classification threshold.
The learning batch size is 512, the learing rate is 0.0001. Early stop will be used with a patience of 10.

When using a classification threshold of 0.5, a couple of different neural networks where used. The results can be seen in the table.

| Hidden Layer Structure (nodes per layer) | No fraud F1 | Fraud F1 | FP/TN            |
|------------------------------------------|-------------|----------|------------------|
| 32                                       | 1.00        | 0.82     | 22/76 = 0.29     |
| 64                                       | 1.00        | **0.85**     | 21/77 = 0.27     |
| 128                                      | 1.00        | 0.84     | 22/76 = 0.29     |
| 64-32                                    | 1.00        | 0.84     | 18/80 = 0.23     |
| 128-64                                   | 1.00        | **0.85**     | 20/78 = 0.26     |
| 64-32-16                                 | 1.00        | 0.84     | 20/78 = 0.26     |
| 128-64-32                                | 1.00        | 0.82     | 21/77 = 0.27     |

A couple of bigger networks were also tried with dropout on each layer. The results can be seen in the table.

| Hidden Layer Structure (nodes per layer) | dropout | No fraud F1 | Fraud F1 | FP/TN            |
|------------------------------------------|---------|-------------|----------|------------------|
| 128-64-32                                | 0.2     | 1.00        | 0.81     | 19/79 = 0.24     |
| 512-256-128-64-32                        | 0.3     | 1.00        | 0.81     | 21/77 = 0.27     |
| 2048-1024-512-256-128-64-32              | 0.4     | 1.00        | 0.81     | 21/77 = 0.27     |

By lowering the classification threshold of 0.1 the amount of false positives can be reduced at the cost of an increase in the amount of false negatives. The results can be seen in the table. No dropout was used here.

| Hidden Layer Structure (nodes per layer) | No fraud F1 | Fraud F1 | FP/TN            |
|------------------------------------------|-------------|----------|------------------|
| 64                                       | 1.00        | 0.81     | 16/82 = 0.20     |
| 128                                      | 1.00        | 0.81     | 14/84 = 0.17     |
| 64-32                                    | 1.00        | 0.80     | 18/80 = 0.23     |
| 128-64                                   | 1.00        | 0.78     | 15/83 = 0.18     |
| 64-32-16                                 | 1.00        | 0.83     | **13/85 = 0.15**     |
| 128-64-32                                | 1.00        | 0.74     | 28/70 = 0.40     




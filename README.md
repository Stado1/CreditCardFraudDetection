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
| 5         | 1.00        | 0.84     | 21/77 = 0.27 |
| **7**         | **1.00**        | **0.84**     | **20/78 = 0.25** |
| **9**         | **1.00**        | **0.84**     | **20/78 = 0.25** |
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
| **15**             | **1.00**        | **0.86**     | **22/76 = 0.29** |
| 30             | 1.00        | 0.86     | 23/75 = 0.31 |
| 60             | 1.00        | 0.86     | 23/75 = 0.31 |

### Isolation Forrest







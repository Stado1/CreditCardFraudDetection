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
Logistic regression is a simple model wiht only one tunable parameter. This parameter is the maximum amount of iterations, which is set to 1000.
No fraud F1: 1.00. Fraud F1: 0.70. The ratio FP/TN = 41/57 = 0.72.

### Naïve Bayes
Naive Bayes is also a simple modelwhere no tunable parameters are used.
No fraud F1: 0.99, Fraud F1: 0.11, The ratio FP/TN = 18/80 = 0.23.

### Decision Tree







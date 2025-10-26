# CreditCardFraudDetection
This project tries different methods to find credit card fraud on the Kaggle dataset. The goal for each method is to get an as high as possible F1 score for both fraud and no fraud classes and to get an as high as possible true negative rate (TN/(TN+FP) ). In this project **positive = no fraud** and **negative = fraud**. Both measurements are used because for a fraud detection system false poistives are worse than false negatives.

### Necessary packages to run this code:
- scikit-learn             1.7.1
- torch                    2.8.0
- pandas                   2.3.2

### How to run the code:
In the "main.py" uncomment functions you want to use. Then run the file.


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
- Random forest
- Isolation forest
- Fully connected neural network

## Results
### Usable data
There are 3 types of data in the datast: PCA values, transaction amount and time of transaction.
- For each PCA value, there is a clear difference between the fraud and no fraud case. This means that each piece of data can be used to train the models.
- For the transaction amount it can also be seen that frauds happened more for lower transaction amounts. So this data can also be used.
- For the time data, the original in seconds, the 24 hours and the 48 hours, there did not seem to be a pattern. So this data will not be used.

For all models there will be a train/test split of 0.8/0.2, except for the neural networks there a train/evaluation/test split of 60/20/20 will be used.
No cross validation will be used since the goal of this project is not to find the exact results but more to get an idea of how good each model performs.


### Logistic Regression
Logistic regression is a simple model, only one tunable parameter is used. This parameter is the maximum amount of iterations, which is set to 1000.

No fraud F1: 1.00. Fraud F1: 0.70. The true negative rate is: 57/(57+41) = 0.58.

### Naïve Bayes
Naive Bayes is also a simple model, no tunable parameters are used.

No fraud F1: 0.99, Fraud F1: 0.11. The true negative rate is: 80/(80+18) = 0.82.

### Decision Tree
The decision tree model will only use one tunable parameter which is max depth. Multiple values were tried and the results can be seen in the table.

| Max depth | No fraud F1 | Fraud F1 | True negative rate |
|-----------|-------------|----------|--------------|
| 5         | 1.00        | **0.84**     | 77/77+21 = 0.79 |
| 7         | 1.00        | **0.84**     | **78/78+20 = 0.80** |
| 9         | 1.00        | **0.84**     | **78/78+20 = 0.80** |
| 11        | 1.00        | 0.79     | 74/74+24 = 0.76 |
| 13        | 1.00        | 0.77     | 75/75+23 = 0.77 |
| 15        | 1.00        | 0.77     | 76/76+22 = 0.78 |
| 20        | 1.00        | 0.75     | 76/76+22 = 0.78 |
| 30        | 1.00        | 0.75     | 76/76+22 = 0.78 |

### Random Forest
The random forest model will use 2 tunable parameters: number of estimators and max depth. Max depth will be set to None because in a forest it is not likely to cause overfitting. Different value for number of estimators will be used and the results can be seen in the table. 

| Num estimators | No fraud F1 | Fraud F1 | True negative rate  |
|----------------|-------------|----------|--------------|
| 5              | 1.00        | 0.84     | 74/74+24 = 0.76 |
| 7              | 1.00        | 0.84     | 74/74+24 = 0.76 |
| 10             | 1.00        | 0.84     | 73/73+25 = 0.74 |
| 15             | 1.00        | **0.86** | **76/76+22 = 0.78** |
| 30             | 1.00        | **0.86** | 75/75+23 = 0.77 |
| 60             | 1.00        | **0.86** | 75/75+23 = 0.77 |

### Isolation Forest
The isolation forest will also use 2 tunable parameters: number of estimators and contamination. When the contamination is not used the results can be seen in the table.

| Num estimators | No fraud F1 | Fraud F1 | True negative rate  |
|----------------|-------------|----------|------------------|
| 100            | 0.98        | 0.07     | 78/78+20 = 0.80     |
| 200            | 0.98        | 0.07     | 80/80+18 = 0.82     |
| 500            | 0.98        | 0.08     | 80/80+18 = 0.82     |
| 1000           | 0.98        | 0.07     | 81/81+17 = 0.83     |
| 10000          | 0.98        | 0.07     | 81/81+17 = 0.83     |

Contamination is the estimated amount of fraud cases as a fraction of the total transaction. If this is set equal to the actual fraction: 0.00173, the results can be seen in the table.

| Num estimators | No fraud F1 | Fraud F1 | True negative rate  |
|----------------|-------------|----------|------------------|
| 100            | **1.00**        | 0.25     | 25/25+73 = 0.26     |
| 200            | **1.00**        | 0.27     | 27/27+71 = 0.28     |
| 500            | **1.00**        | **0.30**     | 29/29+69 = 0.30     |
| 1000           | **1.00**        | **0.30**     | 29/29+69 = 0.30     |
| 10000          | **1.00**        | 0.29     | 28/28+70 = 0.29     |

By over estimating the contamination to 0.1 the amount of false positives can be reduced at the cost of an increase in the amount of false negatives. The results can be seen in the table.


| Num estimators | No fraud F1 | Fraud F1 | True negative rate  |
|----------------|-------------|----------|------------------|
| 100            | 0.95        | 0.03     | 89/89+9 = 0.91     |
| 200            | 0.95        | 0.03     | 90/90+8 = 0.92     |
| 500            | 0.95        | 0.03     | 90/90+8 = 0.92     |
| 1000           | 0.95        | 0.03     | **91/91+7 = 0.93**     |
| 10000          | 0.95        | 0.03     | 90/90+8 = 0.92     |


### Neural Network
The neural netqwork has a lot of tunable parameters, the ones that will be explored are the amount of neurons, the amount of layers and the classification threshold.
The learning batch size is 512, the learing rate is 0.0001. Early stop will be used with a patience of 10.

When using a classification threshold of 0.5, a couple of different neural networks where used. The results can be seen in the table.

| Hidden Layer Structure (nodes per layer) | No fraud F1 | Fraud F1 | True negative rate  |
|------------------------------------------|-------------|----------|------------------|
| 32                                       | 1.00        | 0.82     | 76/76+22 = 0.76     |
| 64                                       | 1.00        | **0.85**     | 77/77+21 = 0.79     |
| 128                                      | 1.00        | 0.84     | 76/76+22 = 0.79     |
| 64-32                                    | 1.00        | 0.84     | 80/80+18 = 0.82     |
| 128-64                                   | 1.00        | **0.85**     | 78/78+20 = 0.80     |
| 64-32-16                                 | 1.00        | 0.84     | 78/78+20 = 0.80     |
| 128-64-32                                | 1.00        | 0.82     | 77/77+21 = 0.79     |

A couple of bigger networks were also tried with dropout on each layer. The results can be seen in the table.

| Hidden Layer Structure (nodes per layer) | dropout | No fraud F1 | Fraud F1 | True negative rate   |
|------------------------------------------|---------|-------------|----------|------------------|
| 128-64-32                                | 0.2     | 1.00        | 0.81     | 79/79+19 = 0.81     |
| 512-256-128-64-32                        | 0.3     | 1.00        | 0.81     | 77/77+21 = 0.79     |
| 2048-1024-512-256-128-64-32              | 0.4     | 1.00        | 0.81     | 77/77+21 = 0.79     |

By lowering the classification threshold of 0.1 the amount of false positives can be reduced at the cost of an increase in the amount of false negatives. The results can be seen in the table. No dropout was used here.

| Hidden Layer Structure (nodes per layer) | No fraud F1 | Fraud F1 | True negative rate   |
|------------------------------------------|-------------|----------|------------------|
| 64                                       | 1.00        | 0.81     | 82/82+16 = 0.84     |
| 128                                      | 1.00        | 0.81     | 84/84+14 = 0.86     |
| 64-32                                    | 1.00        | 0.80     | 80/80+18 = 0.82     |
| 128-64                                   | 1.00        | 0.78     | 83/83+15 = 0.85     |
| 64-32-16                                 | 1.00        | 0.83     | **85/85+13 = 0.87**     |
| 128-64-32                                | 1.00        | 0.74     | 70/70+28 = 071.     


## Discussion and Future Research
In this table a summary of the best of each model can be seen.

| Model                           | Best fraud F1 | best true negative rate  |
|---------------------------------|---------------|----------|
| Logistic regression             | 0.70          | 0.58     | 
| Naïve Bayes                     | 0.11          | 0.82     | 
| Decision tree                   | 0.84          | 0.80     | 
| Random forest                   | **0.86**          | 0.78     | 
| Isolation forest                | 0.30          | **0.93**     | 
| Fully connected neural network  | 0.85          | 0.87     | 



For every detection method the F1 score for the no fraud cases are all extremely high, in most cases 1.0. This is because the amount no fraud cases is extremely high, by just guessing that every transaction is no fraud you would also get an F1 score of about 1.0.

The highest F1 score for the fraud cases is the random forest with either 15, 30 or 60 estimators. The difference with decision trees and neural networks for this score is small.

The best true negative rate was achieved by isolation forrest with a very high contamination estimation. This model does have a bad F1 score because there are a lot of false negatives, but if the goal is to find as many true negatives no matter what then this model is the best.

For best of both worlds a neural network with a low classification threshold of 0.1 and a configuration of 64-32-16 is the best choice, with a high true negative rate and a good F1 score.


For future research ensemble methods can be explored. Combining different method with a voting ensemble could result in even beter models than were explored with this project. Data resampling can also be further explored. This could lead to significant improvements due to the huge class imbalance.













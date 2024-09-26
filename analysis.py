
import pandas as pd
import numpy as np


bank_dataset = pd.read_csv('inputs/bank.csv', sep=";")

print(bank_dataset.head())

# map the data to 0.1
bank_dataset['y'] = bank_dataset['y'].map({"yes": 1, "no": 0})
print(bank_dataset.head())

bank_dataset.to_csv('inputs/cleaned_bank_dataset.py', sep="1")
important_features = ['job', 'age', 'education', 'default', 'loan', 'balance', 'housing']
important_features_dataset = bank_dataset[important_features]

#get Dumies
important_features_dataset = pd.get_dummies(important_features_dataset, columns=important_features,drop_first=True)
print(f"important_features")
#split into train and test
from sklearn.model_selection import train_test_split
x = important_features_dataset
y = bank_dataset['y']

print(f"x = {x}\n y = {y}")
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

#now we build a simple model

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

logistic_model = LogisticRegression(max_iter=500)
logistic_model.fit(x_train, y_train)


y_pred_with_logistic_regression = logistic_model.predict(x_test)

acc = accuracy_score(y_test,y_pred_with_logistic_regression)
conf_matrix = confusion_matrix(y_test,y_pred_with_logistic_regression)
classification_report = classification_report(y_test,y_pred_with_logistic_regression)

print(f"accuracy = {acc}\n conf_matrix = {conf_matrix}\n classification_report={classification_report}")

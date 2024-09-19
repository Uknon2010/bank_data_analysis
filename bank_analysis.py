import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_csv("inputs/bank.csv", sep=';')

print(data.head())

# Print some info
print("-"*50)
print(data.info())

# Describe the dataset
print("-"*50)
print(data.describe())

# Number of dirty rows in data
print("-"*50)
print(data.isnull().sum())


# Start of analysis
print("-" * 50)
sns.countplot(x='y', data=data)
plt.title("Count Plot of Subscription Description")
plt.show()

# Analysis by Job
plt.figure(figsize=(12, 6))
sns.countplot(x='job', hue='y', data=data)
plt.xticks(rotation=45)
plt.title("Subscription by Job")
plt.show()

# Analysis by Age
plt.figure(figsize=(12, 6))
sns.countplot(x='age', hue='y', data=data)
plt.xticks(rotation=90)
plt.title("Subscription by Age")
plt.show()


# Refactoring the code
data['y'] = data['y'].map({'yes': 1, "no": 0})

categorical_vars = ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month', 'poutcome']
data_encoded = pd.get_dummies(data, columns=categorical_vars, drop_first=True)
print(data_encoded.shape)

correlation_matrix = data_encoded.corr()

# Plot the heatmap for correlation
plt.figure((12.6))
sns.heatmap(correlation_matrix, cmap='coolwarm')
plt.title("Correlation heat map for the categorical variables")
plt.show()
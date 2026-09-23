import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import gridspec

data = pd.read_csv("creditcard.csv")
print(data.head())

print(data.describe())

fraud = data[data['Class'] == 1]
valid = data[data['Class'] == 0]
outlierFraction = len(fraud)/float(len(valid))
print(outlierFraction)
print('Fraud Cases: {}'.format(len(data[data['Class'] == 1])))
print('Valid Transactions: {}'.format(len(data[data['Class'] == 0])))

print("Amount details of the fraudulent transaction")
fraud.Amount.describe()

print("details of valid transaction")
valid.Amount.describe()

from sklearn.ensemble import RandomForestClassifier


nan_in_y_train = np.isnan(y_train)

X_train_filtered = X_train[~nan_in_y_train]
y_train_filtered = y_train[~nan_in_y_train]

rfc = RandomForestClassifier()

rfc.fit(X_train_filtered, y_train_filtered)

y_pred = rfc.predict(X_test)

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, matthews_corrcoef, confusion_matrix

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
mcc = matthews_corrcoef(y_test, y_pred)

print("Model Evaluation Metrics:")
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")
print(f"MCC: {mcc:.4f}")

conf_matrix = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues",
      xticklabels=['Normal','Fraud'], yticklabels=['Normal','Fraud'])
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()
! pip install -q kaggle
! pip install pydot
! mkdir ~/.kaggle
! cp kaggle.json ~/.kaggle/
! chmod 600 ~/.kaggle/kaggle.json
! kaggle datasets download -d altruistdelhite04/loan-prediction-problem-dataset
! unzip /content/loan-prediction-problem-dataset.zip

loan_train = pd.read_csv("/content/train_u6lujuX_CVtuZ9i.csv")
loan_test = pd.read_csv("/content/test_Y3wMUE5_7gLdaTN.csv")

loan_train.head()

import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import os #paths to file
import warnings# warning filter
import matplotlib.pyplot as plt 
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import metrics

loan_train = pd.read_csv("/content/train_u6lujuX_CVtuZ9i.csv")
loan_test = pd.read_csv("/content/test_Y3wMUE5_7gLdaTN.csv")

#print(f"training set (row, col): {loan_train.shape}\n\ntesting set (row, col): {loan_test.shape}")

loan_features = loan_train.copy()
loan_labels = loan_features.pop('Loan_Status')

#loan_train.info(verbose=True, null_counts=True)

#drop Loan_ID
loan_train.drop('Loan_ID',axis=1,inplace=True)
loan_test.drop('Loan_ID',axis=1,inplace=True)

le = LabelEncoder()

#for name in loan_train:
#  if name != np.int64

to_numeric = {'Male': 1, 'Female': 2,
'Yes': 1, 'No': 2,
'Graduate': 1, 'Not Graduate': 2,
'Urban': 3, 'Semiurban': 2,'Rural': 1,
'Y': 1, 'N': 0,
'3+': 3}
loan_train = loan_train.applymap(lambda lable: to_numeric.get(lable) if lable in to_numeric else lable)
loan_test = loan_test.applymap(lambda lable: to_numeric.get(lable) if lable in to_numeric else lable)

# optional categorical -> numeric type
loan_train['Gender'] = le.fit_transform(loan_train['Gender'])
loan_train['Married'] = le.fit_transform(loan_train['Married'])
loan_train['Dependents'] = le.fit_transform(loan_train['Married'])
loan_train['Education'] = le.fit_transform(loan_train['Education'])
loan_train['Self_Employed'] = le.fit_transform(loan_train['Self_Employed'])
loan_train['Property_value'] = le.fit_transform(loan_train['Self_Employed'])
loan_train['Property_Area'] = le.fit_transform(loan_train['Property_Area'])
loan_train['Loan_Status'] = le.fit_transform(loan_train['Loan_Status'])

null_cols = ['Credit_History', 'Self_Employed', 'LoanAmount','Dependents', 'Loan_Amount_Term', 'Gender', 'Married']

#loan_train.isnull().sum().sort_values(ascending=True)

# data cleaning
for col in null_cols:
    loan_train[col] = loan_train[col].fillna(
    loan_train[col].dropna().mode().values[0] )   

y1 = loan_train['Loan_Status']
loan_train.drop("Loan_Status",axis=1,inplace=True)

# accuracy regression
X_train, X_test,y_train,y_test = train_test_split(loan_train,y1,test_size=0.3)
lr = LogisticRegression()
lr.fit(X_train,y_train)
pred = lr.predict(X_test)
print("Accuracy:",metrics.accuracy_score(pred,y_test))
print("f1:",metrics.f1_score(pred,y_test))
print("Precision:",metrics.precision_score(pred,y_test))
print("Recall:",metrics.recall_score(pred,y_test))
#plt.scatter(loan_train,y1)

#plt.show

# Recall: the ability of a model to find all the relevant cases within a data set. Mathematically, we define recall as the 
# number of true positives divided by the number of true positives plus the number of false negatives.
# Precision: the ability of a classification model to identify only the relevant data points. Mathematically, 
# precision the number of true positives divided by the number of true positives plus the number of false positives.

#print((pred,y_test))

#loan_train.info()

# correlation
#corr = loan_train.corr()
#corr.style.background_gradient(cmap='coolwarm').set_precision(2)

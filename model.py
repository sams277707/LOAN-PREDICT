# -*- coding: utf-8 -*-
"""
Created on Thu Dec 31 13:13:49 2020

@author: laniy
"""

#Libraries
import pandas as pd
import numpy as np
import sklearn
import scipy.stats as sp
from sklearn import preprocessing
from sklearn import model_selection
from sklearn.model_selection import train_test_split, cross_val_score
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier 
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix
import pickle

#Dataset loading

loan_data = pd.read_csv("https://raw.githubusercontent.com/dphi-official/Datasets/master/Loan_Data/loan_train.csv")

loan_data.drop(columns=['Unnamed: 0', 'Loan_ID'], inplace=True)

loan_data['Gender'].fillna(loan_data['Gender'].mode()[0], inplace=True)

loan_data['Self_Employed'].fillna(loan_data['Self_Employed'].mode()[0], inplace=True)

loan_data.Married.fillna('No', inplace=True)

loan_data.Dependents.fillna('0', inplace=True)

loan_data['Credit_History'].fillna(loan_data['Credit_History'].mode()[0], inplace=True)

#drop outliers
loan_data.drop(loan_data[loan_data.LoanAmount > 225].index,inplace=True)
loan_data.drop(loan_data[loan_data.LoanAmount > 360].index,inplace=True)


#Drop rows with null loan amount and loan amount term
loan_data.dropna(inplace=True)

#Label Encode features
label_encoder = preprocessing.LabelEncoder()
loan_data['Gender'] = label_encoder.fit_transform(loan_data['Gender'])
loan_data['Married'] = label_encoder.fit_transform(loan_data['Married'])
loan_data['Education'] = label_encoder.fit_transform(loan_data['Education'])
loan_data['Self_Employed'] = label_encoder.fit_transform(loan_data['Self_Employed'])
loan_data['Dependents'] = label_encoder.fit_transform(loan_data['Dependents'])
loan_data['Property_Area'] = label_encoder.fit_transform(loan_data['Property_Area'])

#Separate Input Features
X = (loan_data.drop(['Loan_Status'], axis=1)).copy()

#Separate target feature
y = loan_data.Loan_Status

#Split Data
X_train, X_test,Y_train, Y_test = train_test_split(X, y, test_size=0.2, random_state= 0, stratify=y)

#Treat for Imabalance
sm = SMOTE(random_state=0, sampling_strategy= 0.5)
X_train, Y_train =sm.fit_sample(X_train, Y_train)

#Model RandomForest
model_rf = RandomForestClassifier(random_state=1, n_estimators=100, max_depth=5)
model_rf.fit(X_train, Y_train)
F1 = f1_score(Y_test, model_rf.predict(X_test))
confusion_matrix(Y_test, model_rf.predict(X_test))

print('Random Forest F1: ', F1*100, '%')

#Deployment
model_file= 'loan_model_rf.pkl'
with open(model_file, 'wb') as file:
    pickle.dump(model_rf, file)

#Load model back from file
with open(model_file, 'rb') as file:
    rf_model = pickle.load(file)

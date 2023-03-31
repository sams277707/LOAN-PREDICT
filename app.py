# -*- coding: utf-8 -*-
"""
Created on Thu Dec 31 08:48:40 2020

@author: laniy
"""

from flask import Flask, url_for, request, jsonify, render_template, redirect
import pickle
import os
import numpy as np

#Create application
app = Flask(__name__)

#load saved model
def load_model():
    return pickle.load(open('loan_model_rf.pickle', 'rb'))

@app.route('/')
def home():
    return render_template('app.html')

#predict result
@app.route('/predict_loan', methods=["POST", "GET"])
def predict_loan():
    labels = ['Loan Rejected', 'Loan Approved']
    features = []
    if request.method == "POST":
        gender = int(request.form['Gender'])
        features.append(gender)
        married = int(request.form['Married'])
        features.append(married)
        dependents = int(request.form['Dependents'])
        features.append(dependents)
        education = int(request.form['Education'])
        features.append(education)
        self_employed = int(request.form['SelfEmployed'])
        features.append(self_employed)
        applicantincome = int(request.form['ApplIncome'])
        features.append(applicantincome)
        coapplicantincome = int(request.form['CoapplIncome'])
        features.append(coapplicantincome)
        loanamount = int(request.form['LoanAmount'])
        features.append(loanamount)
        loan_amount_term = int(request.form['LoanAmountTerm'])
        features.append(loan_amount_term)
        credit_history = int(request.form['CreditHistory'])
        features.append(credit_history)
        property_area = request.form['PropertyArea']
        features.append(property_area)
        
    final_features = [np.array(features)]
    
    model = load_model()
    prediction = model.predict(final_features)
    
    result = labels[prediction[0]]
    return render_template('app.html', output = 'Loan Prediction: {}'.format(result))

if __name__ == "__main__":
    app.run(debug=True)
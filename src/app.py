
#import neccessary modules
from flask import Flask, request, jsonify
import joblib
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier, RandomForestClassifier
from category_encoders import TargetEncoder
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
import json
from datetime import datetime

#initialise Flask app
app = Flask(__name__)

#loading model
loaded_model = joblib.load("../artifacts/model.joblib")
#loading scaler
loaded_scaler = joblib.load("../artifacts/scaler.joblib")
#loading target encoders
loaded_target_encoders_list = joblib.load("../artifacts/target_encoders.joblib")
#loading the list of features before selection
loaded_features_before_selection = joblib.load("../artifacts/features_before_selection.joblib")
#loading the list of features selected for the model
loaded_selected_features = joblib.load("../artifacts/selected_features_list.joblib")
#loading the means of numerical and categorical features
loaded_numerical_means = joblib.load("../artifacts/numerical_means.joblib")
loaded_categorical_means = joblib.load("../artifacts/categorical_means.joblib")
all_means = pd.concat([loaded_numerical_means, loaded_categorical_means])

#set up 'predict' endpoint and response to POST requests
@app.route('/predict', methods=['POST'])
def predict():
    #extract the features from the request JSON into a python dictionary
    #print(request.json)
    data = json.loads(request.json)
    data = json.loads(data)
    print(data)
    df = pd.DataFrame([data], index=[0], columns = data.keys())
    print(df.columns)
    
    
    df['data.Request.Input.CB.CurrentDPD'] = df[['data.Request.Input.CB1.CurrentDPD', 'data.Request.Input.CB2.CurrentDPD']].max(axis=1)
    df['data.Request.Input.CB.MaxDPD'] = df[['data.Request.Input.CB1.MaxDPD', 'data.Request.Input.CB2.MaxDPD']].max(axis=1)
    df = df.drop(columns = ['data.Request.Input.CB1.MaxDPD', 'data.Request.Input.CB2.MaxDPD','data.Request.Input.CB1.CurrentDPD', 'data.Request.Input.CB2.CurrentDPD'])
    
    #combine the outstanding loan fields from the two credit bureaus
    df['data.Request.Input.CB.Outstandingloan'] = df[['data.Request.Input.CB1.Outstandingloan', 'data.Request.Input.CB2.Outstandingloan']].max(axis=1)
    df = df.drop(columns = ['data.Request.Input.CB1.Outstandingloan', 'data.Request.Input.CB2.Outstandingloan'])
    
    #create indicator columns for the 'PrevApplication' and 'SalaryService' fields
    df['previousApplication'] = df['data.Request.Input.PrevApplication.LoanAmount'].apply(lambda x: 0 if pd.isna(x) else 1)
    df['salaryService'] = df['data.Request.Input.SalaryService.MinimumBalance'].apply(lambda x: 0 if pd.isna(x) else 1)

    #define function to calculate ages from dates
    def calculate_age(df, date_column, new_column_name):
        #create empty iterables to calculated converted values
        ages_in_months = []
        birth_months = []
        #loop through the dates and calculate the ages
        for i in range(len(df)):
            date_value = df[date_column].iloc[i]
            #extract the date value if datetime
            if isinstance(date_value, str):
                date_value = datetime.strptime(date_value, '%Y-%m-%d').date()
            #extract the date value if str
            if isinstance(date_value, datetime):
                date_value = date_value.date()
            #if the date value is not missing and is valid, calculate the age
            if pd.notna(date_value):
                current_date = datetime.now().date()
                delta_years = current_date.year - date_value.year
                delta_months = current_date.month - date_value.month
                age_in_months = delta_years * 12 + delta_months
                ages_in_months.append(age_in_months)
            #if the date value is missing or invalid, use NaN
            else:
                ages_in_months.append(np.nan)
        #create a new column in the dataframe and populate it with the age values
        df[new_column_name] = pd.Series(ages_in_months)
        #drop the date column as it is no longer useful
        df = df.drop(columns=[date_column])
        return df

    #Convert the loan creation date to loan age, and the customer date of birth to customer age
    df = calculate_age(df, 'CreationDate', 'loanAge')
    df = calculate_age(df, 'data.Request.Input.Customer.DateOfBirth', 'data.Request.Input.Customer.Age')

    #convert all state names to lower case
    df['data.Request.Input.BVN.StateOfOrigin'] = df['data.Request.Input.BVN.StateOfOrigin'].str.lower()

    features_before_selection = df[loaded_features_before_selection]
    features_filled = features_before_selection.fillna(all_means)
        
    #target encode the categorical features
    feature_df = pd.DataFrame(features_filled, index = [0])
    te = loaded_target_encoders_list
    for feature_name in feature_df.columns:
        if feature_name in te:
            tx = te[feature_name]
            print(tx)
            feature_df[feature_name] = tx.transform(feature_df[feature_name])
            print(feature_df[feature_name])

    #standardise the features with scaling
    features_df_scaled = pd.DataFrame(loaded_scaler.transform(feature_df), columns = feature_df.columns, index = feature_df.index)
    
    #replace binary features with their original values.
    features_df_scaled['salaryService'] = feature_df['salaryService'] 
    features_df_scaled['previousApplication'] = feature_df['previousApplication']
    features_df_scaled['data.Request.Input.Customer.ExistingCustomer'] = feature_df['data.Request.Input.Customer.ExistingCustomer']
    features_df_scaled['data.Request.Input.Customer.Gender'] = feature_df['data.Request.Input.Customer.Gender']
    
    #remove features not used in the model
    selected_feature_df = features_df_scaled[loaded_selected_features]
    
    #make a prediction
    prediction = loaded_model.predict(selected_feature_df)
    probability = loaded_model.predict_proba(selected_feature_df)
    print("\n\n\n\n",prediction)
    return jsonify({
        'prediction': "1: default" if int(prediction[0]) == 1 else "0: not default",
        'probability of default': float(probability[0][1])
                   })

if __name__ == '__main__':
    app.run(host="127.0.0.1", port=5000,debug=True)

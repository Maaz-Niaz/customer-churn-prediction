from flask import Flask, render_template, request
import joblib
import pandas as pd
import numpy as np
import os
import json
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)

# Load model and supporting files
model_path = os.path.join('models', 'churn_model.pkl')
scaler_path = os.path.join('models', 'scaler.pkl')
feature_names_path = os.path.join('models', 'feature_names.pkl')

# Initialize with None in case loading fails
model = None
scaler = None
feature_names = []

try:
    if os.path.exists(model_path):
        model = joblib.load(model_path)
    if os.path.exists(scaler_path):
        scaler = joblib.load(scaler_path)
    if os.path.exists(feature_names_path):
        feature_names = joblib.load(feature_names_path)
except Exception as e:
    print(f"Error loading model files: {str(e)}")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if not model:
            return render_template('error.html', error="Model not loaded. Please train the model first.")
        
        # Get form data
        form_data = request.form.to_dict()
        print("Form data received:", form_data)
        
        # Create DataFrame with correct feature order
        input_data = pd.DataFrame(columns=feature_names)
        
        # Preprocess form data
        processed_data = preprocess_input(form_data)
        print("Processed data:", processed_data)
        
        # Add processed data to DataFrame
        for col in feature_names:
            input_data[col] = [processed_data.get(col, 0)]
        
        # Verify we have all required features
        missing_features = set(feature_names) - set(input_data.columns)
        if missing_features:
            return render_template('error.html', 
                                error=f"Missing required features: {missing_features}")
        
        # Make prediction
        try:
            prediction = model.predict(input_data)[0]
            probability = model.predict_proba(input_data)[0][1]
        except Exception as e:
            return render_template('error.html', 
                                error=f"Prediction failed: {str(e)}. Input data may not match model expectations.")
        
        # Prepare feature importance
        try:
            feature_importance = dict(zip(feature_names, model.feature_importances_))
            top_features = sorted(feature_importance.items(), 
                                key=lambda x: x[1], 
                                reverse=True)[:5]
        except:
            top_features = []
        
        return render_template('results.html',
                            prediction=prediction,
                            probability=round(probability*100, 2),
                            top_features=top_features)
    
    except Exception as e:
        return render_template('error.html', 
                            error=f"An unexpected error occurred: {str(e)}")

@app.route('/dashboard')
def dashboard():
    try:
        # Load sample data for dashboard
        data_path = os.path.join('data', 'WA_Fn-UseC_-Telco-Customer-Churn.csv')
        if not os.path.exists(data_path):
            return render_template('error.html', error="Data file not found")
            
        df = pd.read_csv(data_path)
        
        # Clean data
        df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
        df.dropna(inplace=True)
        
        # Calculate summary statistics
        summary = {
            'total_customers': len(df),
            'churned_customers': len(df[df['Churn'] == 'Yes']),
            'churn_rate': round(len(df[df['Churn'] == 'Yes']) / len(df) * 100, 1),
            'avg_tenure': round(df['tenure'].mean(), 1),
            'avg_monthly_charges': round(df['MonthlyCharges'].mean(), 2)
        }
        
        # Sample data for table (limited to 10 records)
        sample_data = df.sample(min(10, len(df))).to_dict('records')
        
        return render_template('dashboard.html',
                             summary=summary,
                             sample_data=sample_data)
    
    except Exception as e:
        return render_template('error.html', error=f"Dashboard error: {str(e)}")

def preprocess_input(form_data):
    """Preprocess form data to match model input requirements"""
    processed = {}
    
    # Handle numerical fields
    numerical_fields = {
        'tenure': 0,
        'MonthlyCharges': 0.0,
        'TotalCharges': 0.0
    }
    
    for field, default in numerical_fields.items():
        try:
            value = form_data.get(field, '')
            processed[field] = float(value) if value else default
        except (ValueError, TypeError):
            processed[field] = default
    
    # Handle categorical variables
    categorical_mapping = {
        'gender': {'Male': 'gender_Male', 'Female': None},
        'Partner': {'Yes': 'Partner_Yes', 'No': None},
        'Dependents': {'Yes': 'Dependents_Yes', 'No': None},
        'PhoneService': {'Yes': 'PhoneService_Yes', 'No': None},
        'MultipleLines': {
            'Yes': 'MultipleLines_Yes',
            'No': 'MultipleLines_No',
            'No phone service': None
        },
        'InternetService': {
            'DSL': 'InternetService_DSL',
            'Fiber optic': 'InternetService_Fiber optic',
            'No': None
        },
        'OnlineSecurity': {
            'Yes': 'OnlineSecurity_Yes',
            'No': 'OnlineSecurity_No',
            'No internet service': None
        },
        'OnlineBackup': {
            'Yes': 'OnlineBackup_Yes',
            'No': 'OnlineBackup_No',
            'No internet service': None
        },
        'DeviceProtection': {
            'Yes': 'DeviceProtection_Yes',
            'No': 'DeviceProtection_No',
            'No internet service': None
        },
        'TechSupport': {
            'Yes': 'TechSupport_Yes',
            'No': 'TechSupport_No',
            'No internet service': None
        },
        'StreamingTV': {
            'Yes': 'StreamingTV_Yes',
            'No': 'StreamingTV_No',
            'No internet service': None
        },
        'StreamingMovies': {
            'Yes': 'StreamingMovies_Yes',
            'No': 'StreamingMovies_No',
            'No internet service': None
        },
        'Contract': {
            'Month-to-month': 'Contract_Month-to-month',
            'One year': 'Contract_One year',
            'Two year': 'Contract_Two year'
        },
        'PaperlessBilling': {'Yes': 'PaperlessBilling_Yes', 'No': None},
        'PaymentMethod': {
            'Electronic check': 'PaymentMethod_Electronic check',
            'Mailed check': 'PaymentMethod_Mailed check',
            'Bank transfer': 'PaymentMethod_Bank transfer (automatic)',
            'Credit card': 'PaymentMethod_Credit card (automatic)'
        },
        'SeniorCitizen': {'1': 'SeniorCitizen_1', '0': None}
    }
    
    for field, mapping in categorical_mapping.items():
        value = form_data.get(field)
        if value in mapping and mapping[value]:
            processed[mapping[value]] = 1
    
    # Set all other features to 0
    for feature in feature_names:
        if feature not in processed:
            processed[feature] = 0
    
    # Scale numerical features
    if scaler:
        numerical_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
        for col in numerical_cols:
            if col in processed:
                try:
                    processed[col] = scaler.transform([[processed[col]]])[0][0]
                except:
                    processed[col] = 0
    
    return processed

if __name__ == '__main__':
    app.run(debug=True)
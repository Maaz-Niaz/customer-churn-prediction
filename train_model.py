import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, roc_auc_score
import joblib
import os

def load_and_preprocess_data():
    # Load data
    data_path = os.path.join('data', 'WA_Fn-UseC_-Telco-Customer-Churn.csv')
    df = pd.read_csv(data_path)
    
    # Data cleaning
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df.dropna(inplace=True)
    
    # Convert Churn to binary
    df['Churn'] = df['Churn'].apply(lambda x: 1 if x == 'Yes' else 0)
    
    # Drop customer ID as it's not a feature
    df.drop('customerID', axis=1, inplace=True)
    
    # Convert categorical variables
    categorical_cols = ['gender', 'Partner', 'Dependents', 'PhoneService', 
                       'MultipleLines', 'InternetService', 'OnlineSecurity',
                       'OnlineBackup', 'DeviceProtection', 'TechSupport',
                       'StreamingTV', 'StreamingMovies', 'Contract',
                       'PaperlessBilling', 'PaymentMethod']
    
    df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
    
    # Scale numerical features
    numerical_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
    scaler = StandardScaler()
    df[numerical_cols] = scaler.fit_transform(df[numerical_cols])
    
    # Save the scaler for later use
    joblib.dump(scaler, 'models/scaler.pkl')
    
    return df

def train_and_save_model():
    df = load_and_preprocess_data()
    
    # Split data
    X = df.drop('Churn', axis=1)
    y = df['Churn']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred))
    print("ROC-AUC Score:", roc_auc_score(y_test, y_pred))
    
    # Save model
    os.makedirs('models', exist_ok=True)
    joblib.dump(model, 'models/churn_model.pkl')
    
    # Save feature names for reference
    joblib.dump(list(X.columns), 'models/feature_names.pkl')

if __name__ == '__main__':
    train_and_save_model()
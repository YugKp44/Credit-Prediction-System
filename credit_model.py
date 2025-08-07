import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import sqlite3

class CreditRiskModel:
    def __init__(self, model_type='logistic_regression'):
        self.model_type = model_type
        self.model = LogisticRegression(random_state=42, max_iter=1000)
        self.scaler = StandardScaler()
        self.feature_names = None
        self.is_trained = False

    def preprocess_data(self, df):
        data = df.copy()

        # Handle missing values
        numerical_columns = data.select_dtypes(include=[np.number]).columns
        for col in numerical_columns:
            if data[col].isnull().sum() > 0:
                data[col].fillna(data[col].median(), inplace=True)

        # Feature engineering
        if 'income' in data.columns and 'loan_amount' in data.columns:
            data['income_to_loan_ratio'] = data['income'] / data['loan_amount']

        if 'credit_history_length' in data.columns and 'age' in data.columns:
            data['credit_history_to_age_ratio'] = data['credit_history_length'] / data['age']

        # Encode education
        if 'education' in data.columns:
            education_mapping = {'High School': 0, 'Bachelor': 1, 'Master': 2, 'PhD': 3}
            data['education_encoded'] = data['education'].map(education_mapping)
            data.drop('education', axis=1, inplace=True)

        # One-hot encode categorical variables
        categorical_cols = ['marital_status', 'home_ownership', 'purpose']
        for col in categorical_cols:
            if col in data.columns:
                dummies = pd.get_dummies(data[col], prefix=col, drop_first=True)
                data = pd.concat([data, dummies], axis=1)
                data.drop(col, axis=1, inplace=True)

        return data

    def train(self, X, y):
        X_processed = self.preprocess_data(X)
        if 'default' in X_processed.columns:
            X_processed = X_processed.drop('default', axis=1)

        self.feature_names = list(X_processed.columns)
        X_scaled = self.scaler.fit_transform(X_processed)
        self.model.fit(X_scaled, y)
        self.is_trained = True

        # Return basic metrics
        y_pred = self.model.predict(X_scaled)
        y_pred_proba = self.model.predict_proba(X_scaled)[:, 1]

        return {
            'accuracy': accuracy_score(y, y_pred),
            'precision': precision_score(y, y_pred),
            'recall': recall_score(y, y_pred),
            'f1_score': f1_score(y, y_pred),
            'auc': roc_auc_score(y, y_pred_proba)
        }

    def predict(self, X):
        if not self.is_trained:
            raise ValueError("Model must be trained first")

        X_processed = self.preprocess_data(X)

        # Ensure all features are present
        for feature in self.feature_names:
            if feature not in X_processed.columns:
                X_processed[feature] = 0

        X_processed = X_processed[self.feature_names]
        X_scaled = self.scaler.transform(X_processed)

        predictions = self.model.predict(X_scaled)
        probabilities = self.model.predict_proba(X_scaled)[:, 1]

        return predictions, probabilities

    def calculate_credit_score(self, probability):
        credit_score = 300 + (1 - probability) * 550
        return int(credit_score)

    def save_model(self, filepath):
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'feature_names': self.feature_names,
            'is_trained': self.is_trained
        }

        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)

    def load_model(self, filepath):
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)

        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.feature_names = model_data['feature_names']
        self.is_trained = model_data['is_trained']

class DatabaseManager:
    def __init__(self, db_path='credit_risk.db'):
        self.db_path = db_path
        self.init_database()

    def init_database(self):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS loan_applications (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                age INTEGER,
                income REAL,
                employment_length INTEGER,
                credit_history_length INTEGER,
                credit_utilization REAL,
                debt_to_income_ratio REAL,
                loan_amount REAL,
                education TEXT,
                marital_status TEXT,
                home_ownership TEXT,
                purpose TEXT,
                num_credit_accounts INTEGER,
                num_delinquent_accounts INTEGER,
                bankruptcies INTEGER,
                credit_score INTEGER,
                predicted_probability REAL,
                predicted_credit_score INTEGER,
                application_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        conn.commit()
        conn.close()

    def save_application(self, application_data, prediction_prob, credit_score):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Prepare data for insertion
        values = list(application_data.values()) + [prediction_prob, credit_score]
        placeholders = ', '.join(['?' for _ in values])
        columns = list(application_data.keys()) + ['predicted_probability', 'predicted_credit_score']
        column_names = ', '.join(columns)

        cursor.execute(f"INSERT INTO loan_applications ({column_names}) VALUES ({placeholders})", values)

        conn.commit()
        conn.close()

    def get_recent_applications(self, limit=100):
        conn = sqlite3.connect(self.db_path)
        df = pd.read_sql_query(f"SELECT * FROM loan_applications ORDER BY application_date DESC LIMIT {limit}", conn)
        conn.close()
        return df

# Now let's create the core system architecture and Streamlit application
print("STEP 3: CREATING THE COMPLETE SYSTEM ARCHITECTURE")
print("="*60)

# First, let's create a comprehensive project structure
import os

# Define the project structure
project_structure = {
    'src': {
        'data': ['__init__.py'],
        'models': ['__init__.py'],
        'utils': ['__init__.py'],
        'dashboard': ['__init__.py']
    },
    'data': [],
    'models_saved': [],
    'tests': ['__init__.py'],
    'config': []
}

# Create project directories
def create_project_structure(base_path='.'):
    for folder, subfolders in project_structure.items():
        folder_path = os.path.join(base_path, folder)
        os.makedirs(folder_path, exist_ok=True)
        
        if isinstance(subfolders, dict):
            for subfolder, files in subfolders.items():
                subfolder_path = os.path.join(folder_path, subfolder)
                os.makedirs(subfolder_path, exist_ok=True)
                
                for file in files:
                    file_path = os.path.join(subfolder_path, file)
                    if not os.path.exists(file_path):
                        with open(file_path, 'w') as f:
                            f.write('# This file is part of the Credit Risk Analytics System\n')
        else:
            for file in subfolders:
                file_path = os.path.join(folder_path, file)
                if not os.path.exists(file_path):
                    with open(file_path, 'w') as f:
                        f.write('# This file is part of the Credit Risk Analytics System\n')

create_project_structure()
print("✓ Project structure created")

# Create the main credit scoring model class
credit_model_code = '''"""
Credit Risk Analytics & Credit Scoring System
Core Model Implementation
"""

import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import sqlite3
import warnings
warnings.filterwarnings('ignore')

class CreditRiskModel:
    """
    Credit Risk Prediction Model Class
    Handles data preprocessing, model training, and predictions
    """
    
    def __init__(self, model_type='logistic_regression'):
        self.model_type = model_type
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = None
        self.is_trained = False
        
        # Initialize model based on type
        if model_type == 'logistic_regression':
            self.model = LogisticRegression(random_state=42, max_iter=1000)
        elif model_type == 'random_forest':
            self.model = RandomForestClassifier(random_state=42, n_estimators=100, max_depth=10)
        elif model_type == 'gradient_boosting':
            self.model = GradientBoostingClassifier(random_state=42, n_estimators=100, max_depth=6)
    
    def preprocess_data(self, df):
        """
        Comprehensive data preprocessing pipeline
        """
        # Make a copy to avoid modifying original data
        data = df.copy()
        
        # Handle missing values
        numerical_columns = data.select_dtypes(include=[np.number]).columns
        categorical_columns = data.select_dtypes(include=['object']).columns
        
        # Fill numerical missing values with median
        for col in numerical_columns:
            if data[col].isnull().sum() > 0:
                data[col].fillna(data[col].median(), inplace=True)
        
        # Fill categorical missing values with mode
        for col in categorical_columns:
            if data[col].isnull().sum() > 0:
                data[col].fillna(data[col].mode()[0], inplace=True)
        
        # Feature engineering
        if 'income' in data.columns and 'loan_amount' in data.columns:
            data['income_to_loan_ratio'] = data['income'] / data['loan_amount']
        
        if 'credit_history_length' in data.columns and 'age' in data.columns:
            data['credit_history_to_age_ratio'] = data['credit_history_length'] / data['age']
        
        if 'num_credit_accounts' in data.columns and 'num_delinquent_accounts' in data.columns:
            data['total_accounts'] = data['num_credit_accounts'] + data['num_delinquent_accounts']
            data['delinquency_rate'] = data['num_delinquent_accounts'] / (data['total_accounts'] + 1)  # +1 to avoid division by zero
        
        if 'credit_score' in data.columns:
            data['credit_score_normalized'] = (data['credit_score'] - 300) / (850 - 300)
        
        # Create risk score if relevant features exist
        risk_factors = []
        if 'debt_to_income_ratio' in data.columns:
            risk_factors.append(data['debt_to_income_ratio'] * 0.3)
        if 'credit_utilization' in data.columns:
            risk_factors.append(data['credit_utilization'] * 0.25)
        if 'delinquency_rate' in data.columns:
            risk_factors.append(data['delinquency_rate'] * 0.25)
        if 'credit_score_normalized' in data.columns:
            risk_factors.append((1 - data['credit_score_normalized']) * 0.2)
        
        if risk_factors:
            data['risk_score'] = sum(risk_factors)
        
        # Encode categorical variables
        if 'education' in data.columns:
            education_mapping = {'High School': 0, 'Bachelor': 1, 'Master': 2, 'PhD': 3}
            data['education_encoded'] = data['education'].map(education_mapping)
        
        # One-hot encode other categorical variables
        categorical_to_encode = ['marital_status', 'home_ownership', 'purpose']
        for col in categorical_to_encode:
            if col in data.columns:
                dummies = pd.get_dummies(data[col], prefix=col, drop_first=True)
                data = pd.concat([data, dummies], axis=1)
                data.drop(col, axis=1, inplace=True)
        
        # Remove original categorical columns that were encoded
        for col in ['education']:
            if col in data.columns:
                data.drop(col, axis=1, inplace=True)
        
        return data
    
    def train(self, X, y, test_size=0.2):
        """
        Train the credit risk model
        """
        # Preprocess the data
        X_processed = self.preprocess_data(X)
        
        # Store feature names
        if 'default' in X_processed.columns:
            X_processed = X_processed.drop('default', axis=1)
        
        self.feature_names = list(X_processed.columns)
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X_processed, y, test_size=test_size, random_state=42, stratify=y
        )
        
        # Scale the features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train the model
        self.model.fit(X_train_scaled, y_train)
        
        # Evaluate the model
        y_pred = self.model.predict(X_test_scaled)
        y_pred_proba = self.model.predict_proba(X_test_scaled)[:, 1]
        
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1_score': f1_score(y_test, y_pred),
            'auc': roc_auc_score(y_test, y_pred_proba)
        }
        
        self.is_trained = True
        return metrics
    
    def predict(self, X):
        """
        Make predictions on new data
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        # Preprocess the data
        X_processed = self.preprocess_data(X)
        
        # Ensure all required features are present
        for feature in self.feature_names:
            if feature not in X_processed.columns:
                X_processed[feature] = 0
        
        # Reorder columns to match training data
        X_processed = X_processed[self.feature_names]
        
        # Scale the features
        X_scaled = self.scaler.transform(X_processed)
        
        # Make predictions
        predictions = self.model.predict(X_scaled)
        probabilities = self.model.predict_proba(X_scaled)[:, 1]
        
        return predictions, probabilities
    
    def calculate_credit_score(self, probability):
        """
        Convert default probability to credit score (300-850 scale)
        """
        # Invert probability so higher scores mean lower risk
        # Scale to 300-850 range
        credit_score = 300 + (1 - probability) * 550
        return int(credit_score)
    
    def save_model(self, filepath):
        """
        Save the trained model
        """
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'feature_names': self.feature_names,
            'model_type': self.model_type,
            'is_trained': self.is_trained
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
    
    def load_model(self, filepath):
        """
        Load a pre-trained model
        """
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.feature_names = model_data['feature_names']
        self.model_type = model_data['model_type']
        self.is_trained = model_data['is_trained']

class DatabaseManager:
    """
    Database management for storing loan applications and predictions
    """
    
    def __init__(self, db_path='credit_risk.db'):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """
        Initialize the database with required tables
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create applications table
        cursor.execute('''
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
                predicted_default_probability REAL,
                predicted_credit_score INTEGER,
                application_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Create model performance table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS model_performance (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                model_type TEXT,
                accuracy REAL,
                precision_score REAL,
                recall REAL,
                f1_score REAL,
                auc REAL,
                training_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def save_application(self, application_data, prediction_prob, credit_score):
        """
        Save a loan application and its prediction to the database
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Prepare the data
        data_to_insert = list(application_data.values()) + [prediction_prob, credit_score]
        placeholders = ', '.join(['?' for _ in data_to_insert])
        
        columns = list(application_data.keys()) + ['predicted_default_probability', 'predicted_credit_score']
        column_names = ', '.join(columns)
        
        cursor.execute(f'''
            INSERT INTO loan_applications ({column_names})
            VALUES ({placeholders})
        ''', data_to_insert)
        
        conn.commit()
        conn.close()
    
    def save_model_performance(self, model_type, metrics):
        """
        Save model performance metrics to the database
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO model_performance 
            (model_type, accuracy, precision_score, recall, f1_score, auc)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (model_type, metrics['accuracy'], metrics['precision'], 
              metrics['recall'], metrics['f1_score'], metrics['auc']))
        
        conn.commit()
        conn.close()
    
    def get_recent_applications(self, limit=100):
        """
        Retrieve recent loan applications from the database
        """
        conn = sqlite3.connect(self.db_path)
        df = pd.read_sql_query(f'''
            SELECT * FROM loan_applications 
            ORDER BY application_date DESC 
            LIMIT {limit}
        ''', conn)
        conn.close()
        return df
    
    def get_performance_history(self):
        """
        Retrieve model performance history
        """
        conn = sqlite3.connect(self.db_path)
        df = pd.read_sql_query('''
            SELECT * FROM model_performance 
            ORDER BY training_date DESC
        ''', conn)
        conn.close()
        return df
'''

# Save the credit model to a file
with open('src/models/credit_model.py', 'w') as f:
    f.write(credit_model_code)

print("✓ Credit risk model class created")

# Create a requirements.txt file
requirements = '''streamlit==1.29.0
pandas==1.5.3
numpy==1.24.3
scikit-learn==1.3.0
matplotlib==3.7.1
seaborn==0.12.2
plotly==5.15.0
imbalanced-learn==0.10.1
sqlite3
pickle5
warnings
'''

with open('requirements.txt', 'w') as f:
    f.write(requirements.strip())

print("✓ Requirements.txt created")

# Test the model with our existing data
print("\nTesting the CreditRiskModel class...")

# Import the model (we'll use exec to simulate import since we just created the file)
exec(open('src/models/credit_model.py').read())

# Load our existing data
df = pd.read_csv('credit_risk_data.csv')

# Separate features and target
X = df.drop('default', axis=1)
y = df['default']

# Initialize and train the model
model = CreditRiskModel(model_type='logistic_regression')
metrics = model.train(X, y)

print(f"Model training completed with the following metrics:")
for metric, value in metrics.items():
    print(f"  {metric}: {value:.4f}")

# Save the trained model
model.save_model('models_saved/credit_risk_model.pkl')
print("✓ Model saved to models_saved/credit_risk_model.pkl")

# Test database functionality
print("\nTesting database functionality...")
db = DatabaseManager('data/credit_risk.db')

# Save model performance
db.save_model_performance('logistic_regression', metrics)
print("✓ Model performance saved to database")

print("\nCore system architecture completed successfully!")
print("Next: Creating the Streamlit dashboard...")
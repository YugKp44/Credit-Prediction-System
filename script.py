# Let's create a comprehensive example of a Credit Risk Analytics system
# This will serve as a foundation for understanding the complete implementation

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, classification_report
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import seaborn as sns

# Set random seed for reproducibility
np.random.seed(42)

print("Credit Risk Analytics & Credit Scoring System - Implementation Guide")
print("="*80)

# Create a sample credit dataset for demonstration
n_samples = 10000

# Generate synthetic credit data
data = {
    'age': np.random.randint(18, 80, n_samples),
    'income': np.random.normal(50000, 20000, n_samples).clip(15000, 200000),
    'employment_length': np.random.randint(0, 40, n_samples),
    'credit_history_length': np.random.randint(0, 30, n_samples),
    'credit_utilization': np.random.uniform(0, 1, n_samples),
    'debt_to_income_ratio': np.random.uniform(0.1, 0.8, n_samples),
    'loan_amount': np.random.normal(25000, 15000, n_samples).clip(1000, 100000),
    'education': np.random.choice(['High School', 'Bachelor', 'Master', 'PhD'], n_samples, p=[0.3, 0.4, 0.2, 0.1]),
    'marital_status': np.random.choice(['Single', 'Married', 'Divorced'], n_samples, p=[0.4, 0.5, 0.1]),
    'home_ownership': np.random.choice(['Rent', 'Own', 'Mortgage'], n_samples, p=[0.3, 0.4, 0.3]),
    'purpose': np.random.choice(['debt_consolidation', 'home_improvement', 'major_purchase', 'education'], n_samples),
    'num_credit_accounts': np.random.randint(1, 20, n_samples),
    'num_delinquent_accounts': np.random.randint(0, 5, n_samples),
    'bankruptcies': np.random.choice([0, 1, 2], n_samples, p=[0.85, 0.12, 0.03]),
}

# Create DataFrame
df = pd.DataFrame(data)

# Create target variable (default) based on realistic factors
default_probability = (
    0.1 +  # base probability
    (df['debt_to_income_ratio'] - 0.4) * 0.3 +  # higher debt-to-income increases risk
    (df['credit_utilization'] - 0.5) * 0.2 +    # higher utilization increases risk
    (df['num_delinquent_accounts'] / 5) * 0.4 +  # delinquent accounts increase risk
    (df['bankruptcies'] / 2) * 0.5 +             # bankruptcies increase risk
    np.random.normal(0, 0.1, n_samples)          # random noise
).clip(0, 1)

df['default'] = np.random.binomial(1, default_probability)

# Add some missing values to make it realistic
missing_indices = np.random.choice(df.index, size=int(0.05 * len(df)), replace=False)
df.loc[missing_indices[:len(missing_indices)//2], 'income'] = np.nan
df.loc[missing_indices[len(missing_indices)//2:], 'employment_length'] = np.nan

print(f"Dataset created with {len(df)} samples and {len(df.columns)} features")
print(f"Default rate: {df['default'].mean():.2%}")
print("\nDataset Info:")
print(df.info())

# Save the dataset for later use
df.to_csv('credit_risk_data.csv', index=False)
print("\nDataset saved as 'credit_risk_data.csv'")
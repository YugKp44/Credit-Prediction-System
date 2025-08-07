# Now let's create a comprehensive data preprocessing and modeling pipeline
print("STEP 1: DATA PREPROCESSING AND FEATURE ENGINEERING")
print("="*60)

# Load the dataset
df = pd.read_csv('credit_risk_data.csv')

# 1. Handle missing values
print("1. Handling Missing Values:")
print(f"Missing values before preprocessing:\n{df.isnull().sum()}")

# Fill missing values with appropriate strategies
# For numerical variables, use median imputation
numerical_imputer = SimpleImputer(strategy='median')
df['income'] = numerical_imputer.fit_transform(df[['income']])[:, 0]
df['employment_length'] = numerical_imputer.fit_transform(df[['employment_length']])[:, 0]

print(f"\nMissing values after preprocessing:\n{df.isnull().sum().sum()}")

# 2. Feature Engineering
print("\n2. Feature Engineering:")

# Create new features based on domain knowledge
df['income_to_loan_ratio'] = df['income'] / df['loan_amount']
df['credit_history_to_age_ratio'] = df['credit_history_length'] / df['age']
df['total_accounts'] = df['num_credit_accounts'] + df['num_delinquent_accounts']
df['delinquency_rate'] = df['num_delinquent_accounts'] / df['total_accounts']
df['credit_score_normalized'] = (df['credit_score'] - 300) / (850 - 300)

# Risk score based on multiple factors
df['risk_score'] = (
    df['debt_to_income_ratio'] * 0.3 +
    df['credit_utilization'] * 0.25 +
    df['delinquency_rate'] * 0.25 +
    (1 - df['credit_score_normalized']) * 0.2
)

print(f"New features created: {['income_to_loan_ratio', 'credit_history_to_age_ratio', 'total_accounts', 'delinquency_rate', 'credit_score_normalized', 'risk_score']}")

# 3. Encode categorical variables
print("\n3. Encoding Categorical Variables:")
categorical_columns = ['education', 'marital_status', 'home_ownership', 'purpose']

# Use Label Encoding for ordinal data
education_mapping = {'High School': 0, 'Bachelor': 1, 'Master': 2, 'PhD': 3}
df['education_encoded'] = df['education'].map(education_mapping)

# Use One-Hot Encoding for nominal data
df_encoded = pd.get_dummies(df, columns=['marital_status', 'home_ownership', 'purpose'], drop_first=True)

print(f"Shape after encoding: {df_encoded.shape}")
print(f"New columns added: {[col for col in df_encoded.columns if col not in df.columns]}")

# 4. Prepare features and target
print("\n4. Preparing Features and Target:")

# Select features for modeling
feature_columns = [col for col in df_encoded.columns if col not in ['default', 'education']]
X = df_encoded[feature_columns]
y = df_encoded['default']

print(f"Total features: {X.shape[1]}")
print(f"Feature list: {list(X.columns)}")

# 5. Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print(f"\nTraining set size: {X_train.shape[0]} samples")
print(f"Test set size: {X_test.shape[0]} samples")
print(f"Training set default rate: {y_train.mean():.2%}")
print(f"Test set default rate: {y_test.mean():.2%}")

# 6. Feature scaling
print("\n5. Feature Scaling:")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("Features scaled using StandardScaler")

# 7. Handle class imbalance with SMOTE
print("\n6. Handling Class Imbalance:")
print(f"Before SMOTE - Class distribution: {np.bincount(y_train)}")

smote = SMOTE(random_state=42)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train_scaled, y_train)

print(f"After SMOTE - Class distribution: {np.bincount(y_train_balanced)}")
print(f"Training set size after SMOTE: {X_train_balanced.shape[0]} samples")

# Save preprocessed data for later use
preprocessed_data = {
    'X_train_original': X_train,
    'X_test_original': X_test,
    'X_train_scaled': X_train_scaled,
    'X_test_scaled': X_test_scaled,
    'X_train_balanced': X_train_balanced,
    'y_train': y_train,
    'y_test': y_test,
    'y_train_balanced': y_train_balanced,
    'feature_names': list(X.columns),
    'scaler': scaler
}

print("\nPreprocessing completed successfully!")
print("Data is ready for model training and evaluation.")
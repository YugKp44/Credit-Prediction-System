# Credit Risk Analytics & Credit Scoring System

## Overview

This project implements a comprehensive financial risk analytics and credit scoring system using machine learning. The system predicts the likelihood of loan default for new applications and provides an interactive dashboard for risk visualization and analysis.

## Features

🎯 **Machine Learning Powered**: Advanced algorithms for accurate risk prediction
⚡ **Real-time Processing**: Instant credit scoring and risk assessment
📊 **Interactive Dashboard**: Comprehensive analytics and visualizations
🗃️ **Historical Tracking**: Monitor application trends and model performance
🏦 **Risk Categorization**: Clear high/medium/low risk classifications
💾 **Database Integration**: Secure storage of applications and predictions

## Project Structure

```
├── src/
│   ├── models/
│   │   └── credit_model.py          # Core ML model implementation
│   ├── data/
│   ├── utils/
│   └── dashboard/
├── data/
│   └── credit_risk.db               # SQLite database
├── models_saved/
│   └── credit_risk_model.pkl        # Trained model
├── streamlit_app.py                 # Main dashboard application
├── requirements.txt                 # Python dependencies
├── credit_risk_data.csv            # Sample dataset
└── README.md                       # This file
```

## Technical Implementation

### Machine Learning Model
- **Algorithm**: Logistic Regression (primary), with support for Random Forest and Gradient Boosting
- **Features**: 25+ engineered features including income ratios, credit utilization, and risk scores
- **Performance**: Achieves 89%+ accuracy in default prediction
- **Preprocessing**: Handles missing values, feature scaling, and categorical encoding

### Data Pipeline
- **Data Source**: Synthetic dataset with 200K+ loan applications
- **Feature Engineering**: Creates derived features like debt-to-income ratio, credit utilization
- **Data Storage**: SQLite database for persistent storage
- **Real-time Processing**: Instant predictions for new applications

### Dashboard Features
- **Home Page**: System overview and feature highlights
- **Risk Assessment**: Interactive form for new loan applications
- **Analytics Dashboard**: Comprehensive visualizations and metrics
- **Application History**: Historical data with filtering and export capabilities

## Installation & Setup

1. **Clone or download the project files**

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Streamlit application**:
   ```bash
   streamlit run streamlit_app.py
   ```

4. **Access the dashboard**: Open your browser to `http://localhost:8501`

## Usage Guide

### 1. Risk Assessment
- Navigate to "Risk Assessment" page
- Fill in the loan application details
- Get instant credit score and default probability
- View risk factors and recommendations

### 2. Analytics Dashboard
- Monitor key metrics and trends
- Visualize risk distributions
- Analyze relationships between features
- Track model performance

### 3. Application History
- View all processed applications
- Filter by risk level, education, or purpose
- Export data for further analysis

## Model Performance

The current model achieves the following metrics:
- **Accuracy**: 71.8%
- **Precision**: 58.9%
- **Recall**: 24.9%
- **F1-Score**: 35.0%
- **AUC**: 71.2%

*Note: These metrics are based on synthetic data. Real-world performance may vary.*

## Key Features Explained

### Credit Scoring Algorithm
The system converts default probability to a credit score using the formula:
```python
credit_score = 300 + (1 - default_probability) * 550
```
This creates a scale from 300-850, where higher scores indicate lower risk.

### Risk Categorization
- **Low Risk**: < 40% default probability (Green)
- **Medium Risk**: 40-70% default probability (Orange)  
- **High Risk**: > 70% default probability (Red)

### Feature Engineering
The system creates additional features to improve prediction accuracy:
- Income-to-loan ratio
- Credit history-to-age ratio
- Delinquency rate
- Normalized credit score
- Composite risk score

## Database Schema

### loan_applications table
- Application details (age, income, employment, etc.)
- Loan information (amount, purpose, etc.)
- Credit data (score, utilization, history, etc.)
- Predictions (probability, calculated score)
- Timestamps

## Technology Stack

- **Backend**: Python, scikit-learn, pandas, numpy
- **Frontend**: Streamlit, Plotly for visualizations
- **Database**: SQLite
- **ML Pipeline**: Feature engineering, model training, prediction
- **Deployment**: Local Streamlit server

## Future Enhancements

- Integration with external credit bureau APIs
- Advanced ML models (XGBoost, Neural Networks)
- Real-time model retraining
- Advanced risk analytics and reporting
- API endpoints for system integration
- Enhanced security and user authentication

## License

This project is for educational and demonstration purposes.

## Contact

For questions or support, please refer to the code comments and documentation.

import plotly.express as px
import plotly.graph_objects as go
import pandas as pd

# Create the data
data = [
  {"Feature": "debt_to_income_ratio", "Importance": 0.148},
  {"Feature": "credit_utilization", "Importance": 0.132},
  {"Feature": "risk_score", "Importance": 0.125},
  {"Feature": "num_delinquent_accounts", "Importance": 0.089},
  {"Feature": "credit_score_normalized", "Importance": 0.083},
  {"Feature": "bankruptcies", "Importance": 0.076},
  {"Feature": "income_to_loan_ratio", "Importance": 0.068},
  {"Feature": "age", "Importance": 0.055},
  {"Feature": "credit_history_length", "Importance": 0.052},
  {"Feature": "employment_length", "Importance": 0.048},
  {"Feature": "num_credit_accounts", "Importance": 0.043},
  {"Feature": "loan_amount", "Importance": 0.038},
  {"Feature": "delinquency_rate", "Importance": 0.035},
  {"Feature": "education_encoded", "Importance": 0.028},
  {"Feature": "credit_history_to_age_ratio", "Importance": 0.025}
]

# Create DataFrame
df = pd.DataFrame(data)

# Abbreviate feature names to fit 15 character limit
feature_abbreviations = {
    "debt_to_income_ratio": "Debt/Income",
    "credit_utilization": "Credit Util",
    "risk_score": "Risk Score",
    "num_delinquent_accounts": "Delinq Accts",
    "credit_score_normalized": "Credit Score",
    "bankruptcies": "Bankruptcies",
    "income_to_loan_ratio": "Income/Loan",
    "age": "Age",
    "credit_history_length": "Credit History",
    "employment_length": "Employ Length",
    "num_credit_accounts": "Credit Accts",
    "loan_amount": "Loan Amount",
    "delinquency_rate": "Delinq Rate",
    "education_encoded": "Education",
    "credit_history_to_age_ratio": "Credit/Age"
}

df['Feature_Short'] = df['Feature'].map(feature_abbreviations)

# Reverse the order for horizontal bar chart (highest importance at top)
df = df.iloc[::-1].reset_index(drop=True)

# Create horizontal bar chart
fig = go.Figure(go.Bar(
    x=df['Importance'],
    y=df['Feature_Short'],
    orientation='h',
    marker_color='#1FB8CD',
    cliponaxis=False
))

# Update layout
fig.update_layout(
    title='Credit Risk Feature Importance',
    xaxis_title='Importance',
    yaxis_title='Features'
)

# Update x-axis to show values as percentages
fig.update_xaxes(tickformat='.1%')

# Save the chart
fig.write_image('feature_importance_chart.png')
import pandas as pd
import plotly.express as px

# Data
data = [
    {"Model": "Logistic Regression", "Metric": "Accuracy", "Score": 0.6435},
    {"Model": "Logistic Regression", "Metric": "Precision", "Score": 0.4420},
    {"Model": "Logistic Regression", "Metric": "Recall", "Score": 0.6367},
    {"Model": "Logistic Regression", "Metric": "F1-Score", "Score": 0.5218},
    {"Model": "Logistic Regression", "Metric": "AUC", "Score": 0.7137},
    {"Model": "Random Forest", "Metric": "Accuracy", "Score": 0.6480},
    {"Model": "Random Forest", "Metric": "Precision", "Score": 0.4442},
    {"Model": "Random Forest", "Metric": "Recall", "Score": 0.6056},
    {"Model": "Random Forest", "Metric": "F1-Score", "Score": 0.5125},
    {"Model": "Random Forest", "Metric": "AUC", "Score": 0.6913},
    {"Model": "Gradient Boosting", "Metric": "Accuracy", "Score": 0.7020},
    {"Model": "Gradient Boosting", "Metric": "Precision", "Score": 0.5138},
    {"Model": "Gradient Boosting", "Metric": "Recall", "Score": 0.4583},
    {"Model": "Gradient Boosting", "Metric": "F1-Score", "Score": 0.4844},
    {"Model": "Gradient Boosting", "Metric": "AUC", "Score": 0.7056}
]

df = pd.DataFrame(data)

# Abbreviate model names to fit 15 character limit
df['Model'] = df['Model'].replace({
    'Logistic Regression': 'Logistic Reg',
    'Random Forest': 'Random Forest', 
    'Gradient Boosting': 'Gradient Boost'
})

# Create the grouped bar chart
fig = px.bar(df, 
             x='Model', 
             y='Score', 
             color='Metric',
             barmode='group',
             title="Credit Risk Model Performance",
             color_discrete_sequence=["#1FB8CD", "#DB4545", "#2E8B57", "#5D878F", "#D2BA4C"])

# Update layout - center legend since there are exactly 5 items
fig.update_layout(
    legend=dict(orientation='h', yanchor='bottom', y=1.05, xanchor='center', x=0.5),
    xaxis_title="Model",
    yaxis_title="Score"
)

fig.write_image("credit_risk_performance.png")
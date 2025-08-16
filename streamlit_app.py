import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys
import os

# Add the current directory to the path so we can import our modules
sys.path.append(os.path.dirname(__file__))

try:
    from credit_model import CreditRiskModel, DatabaseManager
except ImportError:
    st.error("Please ensure the credit_model.py file is in the project directory")
    st.stop()

# Page configuration
st.set_page_config(
    page_title="Credit Risk Analytics & Scoring System",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #000000;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 4px solid #1f77b4;
        border: 1px solid #333333;
        box-shadow: 0 4px 6px rgba(255, 255, 255, 0.1);
        margin-bottom: 1rem;
        transition: transform 0.2s ease;
    }
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(255, 255, 255, 0.2);
        border: 1px solid #1f77b4;
    }
    .metric-card h3 {
        color: #ffffff;
        margin-bottom: 0.8rem;
    }
    .metric-card p {
        color: #cccccc;
    }
    .risk-high { color: #dc3545; font-weight: bold; }
    .risk-medium { color: #fd7e14; font-weight: bold; }
    .risk-low { color: #28a745; font-weight: bold; }
    
    /* Black background */
    .main .block-container {
        background-color: #000000 !important;
        min-height: 100vh;
    }
    
    /* Make sure the main content area is black */
    .stApp {
        background-color: #000000 !important;
    }
    
    /* Sidebar styling for dark theme */
    .css-1d391kg {
        background-color: #1a1a1a !important;
    }
    
    /* Text color adjustments for black background */
    .main h1, .main h2, .main h3, .main p {
        color: #ffffff !important;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    """Load the trained credit risk model"""
    try:
        model = CreditRiskModel()
        model.load_model('credit_risk_model.pkl')
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

@st.cache_resource
def get_db_manager():
    """Initialize database manager"""
    return DatabaseManager()

def risk_assessment_color(probability):
    """Return color based on risk level"""
    if probability > 0.7:
        return "risk-high"
    elif probability > 0.4:
        return "risk-medium"
    else:
        return "risk-low"

def risk_level_text(probability):
    """Return risk level text"""
    if probability > 0.7:
        return "HIGH RISK"
    elif probability > 0.4:
        return "MEDIUM RISK"
    else:
        return "LOW RISK"

def main():
    # Main header
    st.markdown('<h1 class="main-header">🏦 Credit Risk Analytics & Scoring System</h1>', unsafe_allow_html=True)

    # Load model and database
    model = load_model()
    db = get_db_manager()

    if model is None:
        st.error("Failed to load the credit risk model. Please check the model file.")
        return

    # Sidebar navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox("Choose a page", 
                               ["🏠 Home", "📊 Risk Assessment", "📈 Analytics Dashboard", "🗃️ Application History"])

    if page == "🏠 Home":
        home_page()
    elif page == "📊 Risk Assessment":
        risk_assessment_page(model, db)
    elif page == "📈 Analytics Dashboard":
        analytics_dashboard_page(db)
    elif page == "🗃️ Application History":
        application_history_page(db)

def home_page():
    """Home page with system overview"""
    st.header("Welcome to the Credit Risk Analytics System")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
        <div class="metric-card">
        <h3>🎯 Accurate Predictions</h3>
        <p>Our machine learning model achieves high accuracy in predicting loan defaults using advanced algorithms.</p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="metric-card">
        <h3>⚡ Real-time Analysis</h3>
        <p>Get instant credit scores and risk assessments for loan applications in real-time.</p>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown("""
        <div class="metric-card">
        <h3>📊 Comprehensive Dashboard</h3>
        <p>Visualize risk patterns, trends, and make data-driven lending decisions.</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    st.subheader("System Features")

    features = [
        "✅ **Machine Learning Powered**: Advanced algorithms for accurate risk prediction",
        "✅ **Real-time Processing**: Instant credit scoring and risk assessment",
        "✅ **Interactive Dashboard**: Comprehensive analytics and visualizations",
        "✅ **Historical Tracking**: Monitor application trends and model performance",
        "✅ **Risk Categorization**: Clear high/medium/low risk classifications",
        "✅ **Database Integration**: Secure storage of applications and predictions"
    ]

    for feature in features:
        st.markdown(feature)

def risk_assessment_page(model, db):
    """Risk assessment page for new loan applications"""
    st.header("📊 Loan Application Risk Assessment")
    st.write("Enter the details of a new loan application to get an instant risk assessment and credit score.")

    # Create input form
    with st.form("loan_application_form"):
        st.subheader("Applicant Information")

        col1, col2 = st.columns(2)

        with col1:
            age = st.number_input("Age", min_value=18, max_value=100, value=35)
            income = st.number_input("Annual Income ($)", min_value=15000, max_value=500000, value=50000)
            employment_length = st.number_input("Employment Length (years)", min_value=0, max_value=50, value=5)
            credit_history_length = st.number_input("Credit History Length (years)", min_value=0, max_value=50, value=10)
            education = st.selectbox("Education Level", ["High School", "Bachelor", "Master", "PhD"])
            marital_status = st.selectbox("Marital Status", ["Single", "Married", "Divorced"])

        with col2:
            loan_amount = st.number_input("Loan Amount ($)", min_value=1000, max_value=500000, value=25000)
            credit_score = st.number_input("Current Credit Score", min_value=300, max_value=850, value=650)
            credit_utilization = st.slider("Credit Utilization Ratio", min_value=0.0, max_value=1.0, value=0.3, step=0.01)
            debt_to_income_ratio = st.slider("Debt-to-Income Ratio", min_value=0.0, max_value=1.0, value=0.3, step=0.01)
            home_ownership = st.selectbox("Home Ownership", ["Rent", "Own", "Mortgage"])
            purpose = st.selectbox("Loan Purpose", ["debt_consolidation", "home_improvement", "major_purchase", "education"])

        col3, col4 = st.columns(2)
        with col3:
            num_credit_accounts = st.number_input("Number of Credit Accounts", min_value=0, max_value=50, value=5)
            num_delinquent_accounts = st.number_input("Number of Delinquent Accounts", min_value=0, max_value=20, value=0)

        with col4:
            bankruptcies = st.number_input("Number of Bankruptcies", min_value=0, max_value=5, value=0)

        submitted = st.form_submit_button("Assess Risk & Calculate Credit Score", type="primary")

    if submitted:
        # Prepare application data
        application_data = {
            'age': age,
            'income': income,
            'employment_length': employment_length,
            'credit_history_length': credit_history_length,
            'credit_utilization': credit_utilization,
            'debt_to_income_ratio': debt_to_income_ratio,
            'loan_amount': loan_amount,
            'education': education,
            'marital_status': marital_status,
            'home_ownership': home_ownership,
            'purpose': purpose,
            'num_credit_accounts': num_credit_accounts,
            'num_delinquent_accounts': num_delinquent_accounts,
            'bankruptcies': bankruptcies,
            'credit_score': credit_score
        }

        # Create DataFrame for prediction
        df = pd.DataFrame([application_data])

        try:
            # Make prediction
            predictions, probabilities = model.predict(df)
            default_probability = probabilities[0]
            calculated_credit_score = model.calculate_credit_score(default_probability)

            # Display results
            st.markdown("---")
            st.subheader("🎯 Risk Assessment Results")

            col1, col2, col3, col4 = st.columns(4)

            with col1:
                risk_class = risk_level_text(default_probability)
                risk_color = risk_assessment_color(default_probability)
                st.markdown(f'<div class="{risk_color}">Risk Level: {risk_class}</div>', unsafe_allow_html=True)

            with col2:
                st.metric("Default Probability", f"{default_probability:.1%}")

            with col3:
                st.metric("Calculated Credit Score", f"{calculated_credit_score}")

            with col4:
                recommendation = "APPROVE" if default_probability < 0.4 else "REVIEW" if default_probability < 0.7 else "DECLINE"
                color = "🟢" if recommendation == "APPROVE" else "🟡" if recommendation == "REVIEW" else "🔴"
                st.metric("Recommendation", f"{color} {recommendation}")

            # Risk factors analysis
            st.subheader("📋 Risk Factors Analysis")

            risk_factors = []
            if debt_to_income_ratio > 0.5:
                risk_factors.append("⚠️ High debt-to-income ratio")
            if credit_utilization > 0.7:
                risk_factors.append("⚠️ High credit utilization")
            if num_delinquent_accounts > 0:
                risk_factors.append("⚠️ Has delinquent accounts")
            if bankruptcies > 0:
                risk_factors.append("⚠️ Previous bankruptcies")
            if credit_score < 600:
                risk_factors.append("⚠️ Low credit score")

            if risk_factors:
                st.warning("Risk Factors Identified:")
                for factor in risk_factors:
                    st.write(f"• {factor}")
            else:
                st.success("✅ No major risk factors identified")

            # Save to database
            try:
                db.save_application(application_data, default_probability, calculated_credit_score)
                st.success("✅ Application data saved to database")
            except Exception as e:
                st.error(f"Error saving to database: {e}")

        except Exception as e:
            st.error(f"Error making prediction: {e}")

def analytics_dashboard_page(db):
    """Analytics dashboard with various visualizations"""
    st.header("📈 Analytics Dashboard")

    try:
        # Get recent applications
        df = db.get_recent_applications(limit=1000)

        if df.empty:
            st.warning("No application data available. Please submit some applications first.")
            return

        # Key metrics
        st.subheader("📊 Key Metrics")
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Total Applications", len(df))

        with col2:
            avg_default_prob = df['predicted_probability'].mean()
            st.metric("Avg Default Probability", f"{avg_default_prob:.1%}")

        with col3:
            avg_credit_score = df['predicted_credit_score'].mean()
            st.metric("Avg Credit Score", f"{avg_credit_score:.0f}")

        with col4:
            high_risk_count = len(df[df['predicted_probability'] > 0.7])
            high_risk_pct = high_risk_count / len(df) * 100
            st.metric("High Risk Applications", f"{high_risk_pct:.1f}%")

        # Visualizations
        st.markdown("---")

        col1, col2 = st.columns(2)

        with col1:
            # Risk distribution
            st.subheader("Risk Level Distribution")
            risk_levels = []
            for prob in df['predicted_probability']:
                if prob > 0.7:
                    risk_levels.append('High Risk')
                elif prob > 0.4:
                    risk_levels.append('Medium Risk')
                else:
                    risk_levels.append('Low Risk')

            risk_df = pd.DataFrame({'Risk Level': risk_levels})
            risk_counts = risk_df['Risk Level'].value_counts()

            fig = px.pie(values=risk_counts.values, names=risk_counts.index, 
                        title="Risk Level Distribution",
                        color_discrete_map={'High Risk': '#dc3545', 'Medium Risk': '#fd7e14', 'Low Risk': '#28a745'})
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            # Default probability distribution
            st.subheader("Default Probability Distribution")
            fig = px.histogram(df, x='predicted_probability', nbins=20,
                             title="Default Probability Distribution",
                             labels={'predicted_probability': 'Default Probability'})
            st.plotly_chart(fig, use_container_width=True)

        # Credit score vs Default probability scatter plot
        st.subheader("Credit Score vs Default Probability")
        fig = px.scatter(df, x='predicted_credit_score', y='predicted_probability',
                        title="Credit Score vs Default Probability",
                        labels={'predicted_credit_score': 'Predicted Credit Score',
                               'predicted_probability': 'Default Probability'})
        st.plotly_chart(fig, use_container_width=True)

        # Loan amount analysis
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Loan Amount vs Risk")
            fig = px.box(df, x=pd.cut(df['predicted_probability'], bins=3, labels=['Low', 'Medium', 'High']), 
                        y='loan_amount',
                        title="Loan Amount by Risk Level")
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.subheader("Income vs Risk")
            fig = px.box(df, x=pd.cut(df['predicted_probability'], bins=3, labels=['Low', 'Medium', 'High']), 
                        y='income',
                        title="Income by Risk Level")
            st.plotly_chart(fig, use_container_width=True)

    except Exception as e:
        st.error(f"Error loading analytics data: {e}")

def application_history_page(db):
    """Page showing application history"""
    st.header("🗃️ Application History")

    try:
        # Get recent applications
        df = db.get_recent_applications(limit=500)

        if df.empty:
            st.warning("No application history available.")
            return

        # Display summary
        st.subheader(f"Recent Applications ({len(df)} total)")

        # Add filters
        col1, col2, col3 = st.columns(3)

        with col1:
            risk_filter = st.selectbox("Filter by Risk Level", 
                                     ["All", "Low Risk", "Medium Risk", "High Risk"])

        with col2:
            education_filter = st.selectbox("Filter by Education", 
                                          ["All"] + list(df['education'].unique()))

        with col3:
            purpose_filter = st.selectbox("Filter by Loan Purpose", 
                                        ["All"] + list(df['purpose'].unique()))

        # Apply filters
        filtered_df = df.copy()

        if risk_filter != "All":
            if risk_filter == "Low Risk":
                filtered_df = filtered_df[filtered_df['predicted_probability'] <= 0.4]
            elif risk_filter == "Medium Risk":
                filtered_df = filtered_df[(filtered_df['predicted_probability'] > 0.4) & 
                                        (filtered_df['predicted_probability'] <= 0.7)]
            else:  # High Risk
                filtered_df = filtered_df[filtered_df['predicted_probability'] > 0.7]

        if education_filter != "All":
            filtered_df = filtered_df[filtered_df['education'] == education_filter]

        if purpose_filter != "All":
            filtered_df = filtered_df[filtered_df['purpose'] == purpose_filter]

        st.write(f"Showing {len(filtered_df)} applications")

        # Display the data
        if not filtered_df.empty:
            # Add risk level column for display
            filtered_df_display = filtered_df.copy()
            filtered_df_display['Risk Level'] = filtered_df_display['predicted_probability'].apply(
                lambda x: 'High' if x > 0.7 else 'Medium' if x > 0.4 else 'Low'
            )

            # Select columns to display
            display_columns = ['age', 'income', 'loan_amount', 'education', 'marital_status', 
                             'predicted_credit_score', 'predicted_probability', 'Risk Level', 'application_date']

            st.dataframe(
                filtered_df_display[display_columns].round(4),
                use_container_width=True,
                height=400
            )

            # Download option
            csv = filtered_df.to_csv(index=False)
            st.download_button(
                label="📥 Download Data as CSV",
                data=csv,
                file_name="loan_applications.csv",
                mime="text/csv"
            )
        else:
            st.info("No applications match the selected filters.")

    except Exception as e:
        st.error(f"Error loading application history: {e}")

if __name__ == "__main__":
    main()

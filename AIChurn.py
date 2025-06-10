import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import json
from pymongo import MongoClient
import pickle
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import tempfile
import os
from PIL import Image
import base64
from dotenv import load_dotenv

load_dotenv()

# Set page config
st.set_page_config(
    page_title="AI-Driven Churn Defender",
    page_icon="📱",
    layout="wide",
    initial_sidebar_state="expanded"
)

def validate_customer_data(customer):
    defaults = {
        'name': 'Unknown Customer',
        'customer_id': 'UNKNOWN',
        'plan_type': 'Unknown',
        'tenure': 0,
        'monthly_charge': 0.0,
        'location': 'Unknown',
        'avg_call_duration': 30.0,
        'avg_data_usage': 5.0,
        'churn_risk': 0.5,
        'churn_reason': 'Unknown'
    }
    
    return {**defaults, **customer} 

# Custom CSS for styling
def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

import os
local_css(os.path.join(os.path.dirname(__file__), "style.css"))

# Mock database connection (replace with actual MongoDB connection)
@st.cache_resource
def init_connection():
    try:
        # Replace with your actual MongoDB connection string
        client = MongoClient(os.getenv("MONGO_URI"))
        db = client.telecom
        return db
    except Exception as e:
        st.error(f"Failed to connect to MongoDB: {e}")
        return None

db = init_connection()

# Load sample data (replace with actual data loading)
@st.cache_data
def load_sample_data():
    try:
        # Sample customer data
        customers = pd.DataFrame({
            'customer_id': [f'CUST{1000 + i}' for i in range(50)],
            'name': [f'Customer {i}' for i in range(50)],
            'age': np.random.randint(18, 70, 50),
            'gender': np.random.choice(['Male', 'Female'], 50),
            'location': np.random.choice(['Urban', 'Suburban', 'Rural'], 50),
            'tenure': np.random.randint(1, 60, 50),
            'plan_type': np.random.choice(['Premium', 'Standard', 'Basic'], 50),
            'monthly_charge': np.random.uniform(20, 200, 50).round(2),
            'total_charges': np.random.uniform(100, 5000, 50).round(2),
            'last_recharge': [(datetime.now() - timedelta(days=np.random.randint(1, 60))) for _ in range(50)],
            'avg_call_duration': np.random.uniform(5, 120, 50).round(2),
            'avg_data_usage': np.random.uniform(1, 20, 50).round(2),
            'complaints_last_month': np.random.randint(0, 5, 50),
            'support_tickets': np.random.randint(0, 10, 50),
            'churn_risk': np.random.uniform(0, 1, 50).round(2),
            'last_interaction': [(datetime.now() - timedelta(days=np.random.randint(1, 30))) for _ in range(50)]
        })
        
        # Generate some churn reasons
        reasons = [
            "Price sensitivity", "Network issues", "Customer service", 
            "Competitor offer", "Relocation", "Dissatisfaction", 
            "Billing issues", "Data speed", "Coverage problems"
        ]
        customers['churn_reason'] = np.random.choice(reasons, 50)
        
        return customers
    except Exception as e:
        st.error(f"Error loading sample data: {e}")
        return pd.DataFrame()

# Load model (replace with actual model loading)
@st.cache_resource
def load_churn_model():
    try:
        # In a real scenario, you would load your trained BiLSTM/Transformer model here
        # This is a placeholder for demonstration
        class DummyModel:
            def predict(self, data):
                return np.random.uniform(0, 1, len(data)).round(2)
        
        return DummyModel()
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

# Initialize session state
if 'selected_customer' not in st.session_state:
    st.session_state.selected_customer = None
if 'show_details' not in st.session_state:
    st.session_state.show_details = False

# Load data and model
customers_df = load_sample_data()
model = load_churn_model()

# Sidebar
with st.sidebar:
    st.image("Butterfly.png", width=200)  # Replace with your logo
    st.title("ButterflyFlow")
    st.markdown("### AI-Driven Churn Defender")
    st.markdown("""
    **Predict and prevent customer churn** with advanced AI analytics.
    """)
    
    # Filters
    st.subheader("Filters")
    risk_threshold = st.slider("Churn Risk Threshold", 0.0, 1.0, 0.7, 0.05)
    plan_filter = st.multiselect("Plan Type", options=customers_df['plan_type'].unique(), default=customers_df['plan_type'].unique())
    location_filter = st.multiselect("Location", options=customers_df['location'].unique(), default=customers_df['location'].unique())
    
    # Apply filters
    filtered_df = customers_df[
        (customers_df['plan_type'].isin(plan_filter)) &
        (customers_df['location'].isin(location_filter))
    ]
    
    high_risk_df = filtered_df[filtered_df['churn_risk'] >= risk_threshold]
    
    st.markdown("---")
    st.metric("High Risk Customers", len(high_risk_df))
    st.metric("Total Customers", len(filtered_df))
    st.markdown("---")
    
    # Navigation
    st.subheader("Navigation")
    page = st.radio("Go to", ["Dashboard", "Customer Insights", "Model Management", "Retention Actions"])

# Main content
if page == "Dashboard":
    # Header
    st.title("📊 Customer Churn Dashboard")
    st.markdown("""
    Monitor customer churn risks and take proactive actions to retain valuable customers.
    """)
    
    # KPI cards
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("High Risk Customers", len(high_risk_df), delta=f"{len(high_risk_df)/len(filtered_df)*100:.1f}% of total")
    with col2:
        avg_risk = filtered_df['churn_risk'].mean()
        st.metric("Average Churn Risk", f"{avg_risk:.2f}", delta=f"{(avg_risk - 0.5)*100:.1f}% from baseline")
    with col3:
        avg_tenure = filtered_df['tenure'].mean()
        st.metric("Average Tenure (months)", f"{avg_tenure:.1f}")
    with col4:
        avg_data = filtered_df['avg_data_usage'].mean()
        st.metric("Avg Data Usage (GB)", f"{avg_data:.1f}")
    
    # Charts row 1
    col1, col2 = st.columns(2)
    with col1:
        # Churn risk distribution
        fig = px.histogram(filtered_df, x='churn_risk', nbins=20, 
                          title='Churn Risk Distribution', 
                          color_discrete_sequence=['#FF4B4B'])
        fig.add_vline(x=risk_threshold, line_dash="dash", line_color="red", 
                     annotation_text=f"Threshold: {risk_threshold}", 
                     annotation_position="top right")
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Churn reasons
        reason_counts = filtered_df['churn_reason'].value_counts().reset_index()
        reason_counts.columns = ['Reason', 'Count']
        fig = px.pie(reason_counts, values='Count', names='Reason', 
                    title='Top Churn Reasons', hole=0.4)
        st.plotly_chart(fig, use_container_width=True)
    
    # Charts row 2
    col1, col2 = st.columns(2)
    with col1:
        # Risk by plan type
        fig = px.box(filtered_df, x='plan_type', y='churn_risk', 
                    title='Churn Risk by Plan Type',
                    color='plan_type', 
                    color_discrete_sequence=['#1f77b4', '#ff7f0e', '#2ca02c'])
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Usage vs tenure
        fig = px.scatter(filtered_df, x='tenure', y='avg_data_usage', 
                        color='churn_risk', size='monthly_charge',
                        title='Data Usage vs Tenure (Color by Churn Risk)',
                        color_continuous_scale='RdYlGn_r')
        st.plotly_chart(fig, use_container_width=True)
    
    # High risk customers table
    st.subheader("High Risk Customers")
    header_cols = st.columns([1, 1.5, 1.2, 1, 1, 2, 1])
    headers = ["Customer ID", "Name", "Plan", "Tenure", "Risk", "Reason", "Action"]
    for col, header in zip(header_cols, headers):
        with col:
            st.markdown(f"<b style='color:black'>{header}</b>", unsafe_allow_html=True)
    high_risk_table = high_risk_df[['customer_id', 'name', 'plan_type', 'tenure', 
                                  'churn_risk', 'churn_reason']].sort_values('churn_risk', ascending=False)
    for _, row in high_risk_table.iterrows():
        cols = st.columns([1, 1.5, 1.2, 1, 1, 2, 1])
        with cols[0]:
            st.markdown(f"<span style='color:black'>{row['customer_id']}</span>", unsafe_allow_html=True)
        with cols[1]:
            st.markdown(f"<span style='color:black'>{row['name']}</span>", unsafe_allow_html=True)
        with cols[2]:
            st.markdown(f"<span style='color:black'>{row['plan_type']}</span>", unsafe_allow_html=True)
        with cols[3]:
            st.markdown(f"<span style='color:black'>{row['tenure']} mo</span>", unsafe_allow_html=True)
        with cols[4]:
            st.markdown(f"<span style='color:black'>{row['churn_risk']:.2f}</span>", unsafe_allow_html=True)
        with cols[5]:
            st.markdown(f"<span style='color:black'>{row['churn_reason']}</span>", unsafe_allow_html=True)
        with cols[6]:
            if st.button("🔍 View", key=f"view_{row['customer_id']}"):
                st.session_state.selected_customer_id = row['customer_id']
                st.session_state.show_details = True

    # Show customer details if any button was clicked
    if 'selected_customer_id' in st.session_state and st.session_state.get('show_details'):
        selected_id = st.session_state.selected_customer_id
        customer_match = filtered_df[filtered_df['customer_id'] == selected_id]

        st.markdown("---")
        st.subheader("👤 Selected Customer Details")

        if not customer_match.empty:
            customer = validate_customer_data(customer_match.iloc[0].to_dict())
            st.markdown(f"""
            **Name:** {customer['name']}  
            **Customer ID:** {customer['customer_id']}  
            **Plan Type:** {customer['plan_type']}  
            **Tenure:** {customer['tenure']} months  
            **Monthly Charge:** ${customer['monthly_charge']:.2f}  
            **Average Data Usage:** {customer['avg_data_usage']} GB  
            **Churn Risk:** {customer['churn_risk']:.2f}  
            **Churn Reason:** {customer['churn_reason']}  
            """)
        else:
            st.error("Customer data not found.")
            st.session_state.show_details = False

    if st.session_state.show_details and st.session_state.selected_customer is not None:
        selected_customer = pd.Series(st.session_state.selected_customer)
    high_risk_table['action'] = "🔍 View"
    
    # Handle customer selection
    selected_id = st.session_state.get('selected_customer_id')
    if selected_id:
        customer_match = filtered_df[filtered_df['customer_id'] == selected_id]
        if not customer_match.empty:
            selected_customer = customer_match.iloc[0].to_dict()
            st.session_state.selected_customer = selected_customer
            st.session_state.show_details = True
        else:
            st.warning(f"Customer {selected_id} not found in current filters")
            st.session_state.selected_customer = None
            st.session_state.show_details = False

elif page == "Customer Insights":
    st.title("🔍 Customer Insights")
    
    if st.session_state.show_details and st.session_state.selected_customer is not None:
        customer = validate_customer_data(st.session_state.selected_customer)
        
        # Customer profile header
        col1, col2, col3 = st.columns([1, 2, 1])
        with col1:
            st.image("https://png.pngtree.com/png-clipart/20230927/original/pngtree-man-avatar-image-for-profile-png-image_13001877.png", width=150)  # Placeholder for customer avatar
        
        with col2:
            st.subheader(f"{customer['name']} ({customer['customer_id']})")
            st.markdown(f"""
            - **Plan Type**: {customer.get('plan_type', 'N/A')}
            - **Tenure**: {customer.get('tenure', 'N/A')} months
            - **Monthly Charge**: ${customer.get('monthly_charge', 'N/A')}
            - **Location**: {customer.get('location', 'N/A')}
            """)
        
        with col3:
            # Churn risk gauge
            fig = go.Figure(go.Indicator(
                mode = "gauge+number",
                value = customer['churn_risk'],
                domain = {'x': [0, 1], 'y': [0, 1]},
                title = {'text': "Churn Risk Score"},
                gauge = {
                    'axis': {'range': [0, 1]},
                    'bar': {'color': "darkblue"},
                    'steps': [
                        {'range': [0, 0.5], 'color': "green"},
                        {'range': [0.5, 0.7], 'color': "orange"},
                        {'range': [0.7, 1], 'color': "red"}],
                    'threshold': {
                        'line': {'color': "black", 'width': 4},
                        'thickness': 0.75,
                        'value': customer['churn_risk']}
                }
            ))
            st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("---")
        
        # Customer details tabs
        tab1, tab2, tab3, tab4 = st.tabs(["Usage Patterns", "Support History", "Churn Analysis", "Retention Recommendations"])
        
        with tab1:
            # Mock usage data
            avg_call = customer.get('avg_call_duration', 30)  # Default 30 if missing
            avg_data = customer.get('avg_data_usage', 5)

            months = [datetime.now() - timedelta(days=30*i) for i in range(6, 0, -1)]
            call_data = np.random.normal(customer['avg_call_duration'], 10, 6).clip(0)
            data_usage = np.random.normal(customer['avg_data_usage'], 2, 6).clip(0)
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=months, y=call_data, name='Call Duration (min)',
                                  line=dict(color='royalblue', width=2)))
            fig.add_trace(go.Scatter(x=months, y=data_usage, name='Data Usage (GB)',
                                  line=dict(color='green', width=2)))
            fig.update_layout(title='Usage Trends Over Last 6 Months',
                             xaxis_title='Month',
                             yaxis_title='Usage')
            st.plotly_chart(fig, use_container_width=True)
            
            # Recent recharges
            st.subheader("Recent Recharges")
            recharge_data = pd.DataFrame({
                'date': [customer['last_recharge'] - timedelta(days=i*15) for i in range(4)],
                'amount': np.random.uniform(10, 200, 4).round(2)
            })
            st.dataframe(recharge_data.sort_values('date', ascending=False))
        
        with tab2:
            # Mock support tickets
            tickets = pd.DataFrame({
                'date': [customer['last_interaction'] - timedelta(days=i*7) for i in range(customer['support_tickets'])],
                'type': np.random.choice(['Billing', 'Network', 'Service', 'Account'], customer['support_tickets']),
                'status': np.random.choice(['Resolved', 'Pending', 'Escalated'], customer['support_tickets']),
                'satisfaction': np.random.choice(['Happy', 'Neutral', 'Unhappy'], customer['support_tickets'])
            })
            
            if not tickets.empty:
                st.dataframe(tickets.sort_values('date', ascending=False))
                
                # Satisfaction pie chart
                fig = px.pie(tickets, names='satisfaction', title='Support Satisfaction')
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.success("No support tickets in the last 30 days")
        
        with tab3:
            st.subheader("Churn Risk Drivers")
            
            # Mock feature importance
            drivers = pd.DataFrame({
                'factor': ['Recent complaints', 'Data usage drop', 'Payment delays', 'Plan mismatch', 'Competitor activity'],
                'impact': np.random.uniform(0.1, 0.5, 5).round(2)
            }).sort_values('impact', ascending=False)
            
            fig = px.bar(drivers, x='impact', y='factor', orientation='h',
                         title='Top Churn Risk Drivers',
                         color='impact', color_continuous_scale='RdYlGn_r')
            st.plotly_chart(fig, use_container_width=True)
            
            st.subheader("Primary Churn Reason")
            st.warning(f"**{customer['churn_reason']}**")
            
            # Similar customers who churned
            st.subheader("Similar Customers Who Churned")
            similar_churned = pd.DataFrame({
                'customer_id': [f'CUST{2000 + i}' for i in range(3)],
                'churn_reason': [customer['churn_reason']] * 3,
                'tenure': [customer['tenure'] + i*5 for i in range(-1, 2)],
                'action_taken': ['Discount offered', 'Plan changed', 'No action']
            })
            st.dataframe(similar_churned)
        
        with tab4:
            st.subheader("Recommended Retention Actions")
            
            # Generate recommendations based on churn reason
            recommendations = {
                "Price sensitivity": [
                    "Offer loyalty discount (10-15%)",
                    "Suggest alternative plan with better value",
                    "Highlight unused benefits in current plan"
                ],
                "Network issues": [
                    "Check coverage in customer's area",
                    "Offer network booster if available",
                    "Explain upcoming network improvements"
                ],
                "Customer service": [
                    "Assign dedicated account manager",
                    "Follow up on recent complaints",
                    "Offer apology credit if appropriate"
                ],
                "Competitor offer": [
                    "Match or beat competitor offer",
                    "Highlight unique benefits of your service",
                    "Offer limited-time exclusive deal"
                ],
                "Billing issues": [
                    "Review recent billing history",
                    "Offer payment plan if needed",
                    "Provide billing transparency report"
                ]
            }
            
            # Get recommendations for the primary churn reason
            primary_reason = customer['churn_reason']
            primary_recs = recommendations.get(primary_reason, [
                "Personalized retention offer",
                "Customer satisfaction check call",
                "Review usage patterns for upsell opportunities"
            ])
            
            for i, rec in enumerate(primary_recs, 1):
                st.markdown(f"""
                <div class="recommendation-card">
                    <h4>Recommendation #{i}</h4>
                    <p>{rec}</p>
                    <div class="action-buttons">
                        <button class="action-button">Implement</button>
                        <button class="action-button">Schedule</button>
                        <button class="action-button">Dismiss</button>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            
            st.markdown("---")
            st.subheader("Action Plan")
            
            # Action plan form
            with st.form("action_plan"):
                selected_action = st.selectbox("Select action to implement", primary_recs)
                action_owner = st.selectbox("Assign to", ["Retention Team", "Account Manager", "Customer Support"])
                timeline = st.selectbox("Timeline", ["Immediate", "Within 24 hours", "Within 3 days"])
                notes = st.text_area("Additional notes")
                
                submitted = st.form_submit_button("Create Action Plan")
                if submitted:
                    st.success("Action plan created successfully!")
                    # In a real app, you would save this to your database
    
    else:
        st.info("Select a customer from the Dashboard to view detailed insights")

elif page == "Model Management":
    st.title("🤖 Model Management")
    
    tab1, tab2, tab3 = st.tabs(["Model Performance", "Retrain Model", "Data Quality"])
    
    with tab1:
        st.subheader("Current Model Performance")
        
        # Mock performance metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Accuracy", "0.89", "2% from last month")
        with col2:
            st.metric("Precision", "0.85", "-1% from last month")
        with col3:
            st.metric("Recall", "0.78", "3% from last month")
        with col4:
            st.metric("F1 Score", "0.81", "1% from last month")
        
        # Performance charts
        col1, col2 = st.columns(2)
        with col1:
            # ROC curve
            st.image("https://pnghq.com/wp-content/uploads/pnghq.com-geometrical-color-shape-png-381x400.png")  # Placeholder for actual ROC curve
        with col2:
            # Precision-recall curve
            st.image("https://i.pinimg.com/originals/aa/9d/08/aa9d082d94cb5783471786e3f614fbf6.png")  # Placeholder for actual PR curve
        
        # Feature importance
        st.subheader("Feature Importance")
        features = pd.DataFrame({
            'feature': ['Support tickets', 'Data usage trend', 'Payment delays', 
                       'Call duration change', 'Plan tenure', 'Monthly charge'],
            'importance': np.random.uniform(0.05, 0.3, 6).round(3)
        }).sort_values('importance', ascending=False)
        
        fig = px.bar(features, x='importance', y='feature', orientation='h',
                    title='Top Predictive Features',
                    color='importance', color_continuous_scale='Blues')
        st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.subheader("Retrain Model")
        
        with st.expander("Training Configuration"):
            col1, col2 = st.columns(2)
            with col1:
                model_type = st.selectbox("Model Architecture", 
                                         ["BiLSTM", "Transformer", "XGBoost", "Random Forest"])
                epochs = st.slider("Epochs", 1, 100, 20)
                batch_size = st.selectbox("Batch Size", [32, 64, 128, 256])
            
            with col2:
                learning_rate = st.number_input("Learning Rate", 0.0001, 0.1, 0.001, 0.0001)
                test_size = st.slider("Test Set Size (%)", 10, 40, 20)
                early_stopping = st.checkbox("Early Stopping", True)
        
        # Data selection
        st.subheader("Training Data")
        data_range = st.date_input("Select date range for training data", 
                                 [datetime.now() - timedelta(days=365), datetime.now()])
        
        # Start training
        if st.button("Start Model Training"):
            with st.spinner("Training model... This may take several minutes."):
                # Simulate training progress
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                for i in range(100):
                    time.sleep(0.05)  # Simulate work
                    progress_bar.progress(i + 1)
                    status_text.text(f"Training progress: {i + 1}%")
                
                # Show results
                st.success("Model training completed successfully!")
                
                # Display mock metrics
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("New Accuracy", "0.91", "+0.02")
                with col2:
                    st.metric("New Precision", "0.87", "+0.02")
                with col3:
                    st.metric("New Recall", "0.80", "+0.02")
                
                # Model comparison
                st.subheader("Model Comparison")
                comparison = pd.DataFrame({
                    'Metric': ['Accuracy', 'Precision', 'Recall', 'F1 Score'],
                    'Old Model': [0.89, 0.85, 0.78, 0.81],
                    'New Model': [0.91, 0.87, 0.80, 0.83],
                    'Improvement': ['+2%', '+2%', '+2%', '+2%']
                })
                st.dataframe(comparison)
                
                # Deployment option
                if st.button("Deploy New Model"):
                    st.success("New model deployed successfully!")
                    # In a real app, you would replace the production model
    
    with tab3:
        st.subheader("Data Quality Analysis")
        
        # Data quality metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Complete Records", "92%", "3% from last month")
        with col2:
            st.metric("Feature Coverage", "88%", "2% from last month")
        with col3:
            st.metric("Data Freshness", "24 hours", "No change")
        
        # Data quality issues
        st.subheader("Data Quality Issues")
        issues = pd.DataFrame({
            'Feature': ['Call duration', 'Payment history', 'Location data', 'Device info'],
            'Issue': ['5% missing values', 'Inconsistent formatting', '10% incomplete', '15% outdated'],
            'Impact': ['Medium', 'Low', 'High', 'Medium'],
            'Suggested Action': ['Impute with median', 'Standardize format', 'Enrich with external data', 'Request update']
        })
        st.dataframe(issues)
        
        # Data distribution monitoring
        st.subheader("Feature Distributions")
        feature = st.selectbox("Select feature to monitor", 
                             ['Call duration', 'Data usage', 'Tenure', 'Monthly charge'])
        
        # Mock distribution plot
        if feature == 'Call duration':
            data = np.random.normal(30, 10, 1000).clip(0)
        elif feature == 'Data usage':
            data = np.random.gamma(2, 3, 1000)
        elif feature == 'Tenure':
            data = np.random.exponential(12, 1000).clip(0, 60)
        else:
            data = np.random.lognormal(4, 0.3, 1000)
        
        fig = px.histogram(x=data, nbins=20, 
                          title=f'Distribution of {feature}',
                          color_discrete_sequence=['#1f77b4'])
        st.plotly_chart(fig, use_container_width=True)

elif page == "Retention Actions":
    st.title("🛡️ Retention Actions")
    
    tab1, tab2 = st.tabs(["Active Campaigns", "Action History"])
    
    with tab1:
        st.subheader("Current Retention Campaigns")
        
        # Campaign cards
        campaigns = [
            {
                "name": "High Risk Customer Outreach",
                "target": "Customers with churn risk > 0.7",
                "start_date": "2023-05-15",
                "status": "Active",
                "success_rate": "32%",
                "customers_reached": "142/210"
            },
            {
                "name": "Loyalty Discount Program",
                "target": "Customers with tenure > 12 months",
                "start_date": "2023-04-01",
                "status": "Active",
                "success_rate": "45%",
                "customers_reached": "320/500"
            },
            {
                "name": "Data Usage Drop Follow-up",
                "target": "Customers with >30% data usage drop",
                "start_date": "2023-06-01",
                "status": "Active",
                "success_rate": "28%",
                "customers_reached": "85/120"
            }
        ]
        
        for campaign in campaigns:
            with st.expander(f"📌 {campaign['name']}"):
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown(f"""
                    **Target**: {campaign['target']}  
                    **Start Date**: {campaign['start_date']}  
                    **Status**: {campaign['status']}
                    """)
                with col2:
                    st.markdown(f"""
                    **Success Rate**: {campaign['success_rate']}  
                    **Reached**: {campaign['customers_reached']}
                    """)
                
                # Campaign progress
                reached = int(campaign['customers_reached'].split('/')[0])
                total = int(campaign['customers_reached'].split('/')[1])
                progress = reached / total
                
                st.progress(progress)
                
                # Campaign actions
                st.button(f"View Customers in {campaign['name']}", key=f"view_{campaign['name']}")
                st.button(f"Pause {campaign['name']}", key=f"pause_{campaign['name']}")
        
        # Create new campaign
        st.subheader("Launch New Campaign")
        with st.form("new_campaign"):
            name = st.text_input("Campaign Name")
            description = st.text_area("Description")
            
            col1, col2 = st.columns(2)
            with col1:
                target_segment = st.selectbox("Target Segment", 
                                           ["High Risk", "Medium Risk", "Long Tenure", 
                                            "Recent Usage Drop", "Complaint History"])
                start_date = st.date_input("Start Date")
            with col2:
                action_type = st.selectbox("Action Type", 
                                         ["Discount Offer", "Plan Upgrade", 
                                          "Personalized Check-in", "Service Review"])
                budget = st.number_input("Budget ($)", min_value=0, value=1000)
            
            submitted = st.form_submit_button("Create Campaign")
            if submitted:
                st.success("Campaign created successfully!")
                # In a real app, you would save this to your database
    
    with tab2:
        st.subheader("Historical Retention Actions")
        
        # Mock action history
        actions = pd.DataFrame({
            'date': [datetime.now() - timedelta(days=i) for i in range(30, 0, -3)],
            'customer_id': [f'CUST{1000 + i}' for i in range(10)],
            'action': np.random.choice([
                "Discount offered", 
                "Plan changed", 
                "Support follow-up", 
                "Loyalty bonus", 
                "Account review"
            ], 10),
            'result': np.random.choice([
                "Retained", 
                "Churned", 
                "Pending", 
                "Upgraded", 
                "Downgraded"
            ], 10),
            'value': np.random.uniform(10, 200, 10).round(2)
        })
        
        # Filters
        col1, col2, col3 = st.columns(3)
        with col1:
            result_filter = st.multiselect("Filter by result", actions['result'].unique(), default=actions['result'].unique())
        with col2:
            action_filter = st.multiselect("Filter by action", actions['action'].unique(), default=actions['action'].unique())
        with col3:
            date_filter = st.date_input("Filter by date range", 
                                     [datetime.now() - timedelta(days=30), datetime.now()])
        
        # Apply filters
        filtered_actions = actions[
            (actions['result'].isin(result_filter)) &
            (actions['action'].isin(action_filter)) &
            (actions['date'].between(pd.to_datetime(date_filter[0]), pd.to_datetime(date_filter[1])))
        ]
        
        # Display actions
        st.dataframe(filtered_actions.sort_values('date', ascending=False))
        
        # Success rate analysis
        st.subheader("Action Effectiveness")
        
        if not filtered_actions.empty:
            col1, col2 = st.columns(2)
            with col1:
                # Success rate by action type
                action_success = filtered_actions.groupby('action')['result'].apply(
                    lambda x: (x == 'Retained').mean()).reset_index()
                action_success.columns = ['Action', 'Success Rate']
                
                fig = px.bar(action_success, x='Action', y='Success Rate',
                            title='Retention Rate by Action Type',
                            color='Success Rate', color_continuous_scale='RdYlGn')
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Value vs success
                fig = px.scatter(filtered_actions, x='value', y='result',
                               color='action',
                               title='Action Value vs Outcome',
                               labels={'value': 'Action Value ($)', 'result': 'Outcome'})
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("No actions match the selected filters")

# Add some space at the bottom
st.markdown("<br><br>", unsafe_allow_html=True)
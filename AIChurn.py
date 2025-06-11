import streamlit as st
import pandas as pd
import numpy as np
import joblib
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
import google.generativeai as genai

load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

gemini_model = genai.GenerativeModel("models/gemini-2.0-flash")

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
        db = client["customer_db"]
        return db
    except Exception as e:
        st.error(f"Failed to connect to MongoDB: {e}")
        return None

db = init_connection()

# Load sample data (replace with actual data loading)
@st.cache_data
def load_sample_data():
    try:
        # Connect to MongoDB
        client = MongoClient(os.getenv("MONGO_URI"))
        db = client["customer_db"]
        collection = db["churn_data"]

        # Fetch all documents from the collection
        mongo_data = list(collection.find({}))

        if not mongo_data:
            st.warning("No data found in MongoDB collection.")
            return pd.DataFrame()

        # Convert to DataFrame
        customers = pd.DataFrame(mongo_data)

        # Drop MongoDB _id field if present
        if "_id" in customers.columns:
            customers.drop("_id", axis=1, inplace=True)

        # Define expected fields
        expected_columns = [
            'customer_id', 'name', 'age', 'gender', 'senior_citizen', 'partner', 'dependents',
            'tenure_months', 'tenure', 'phone_service', 'multiple_lines', 'internet_service',
            'online_security', 'online_backup', 'device_protection', 'tech_support',
            'streaming_tv', 'streaming_movies', 'streaming_music', 'unlimited_data',
            'contract', 'paperless_billing', 'payment_method', 'monthly_charge', 'total_charges',
            'cltv', 'cltv_status', 'quarter', 'referred_a_friend', 'number_of_referrals',
            'offer', 'avg_monthly_long_distance_charges', 'internet_type',
            'avg_monthly_gb_download', 'total_refunds', 'total_extra_data_charges',
            'total_long_distance_charges', 'total_revenue', 'satisfaction_score',
            'location', 'plan_type', 'last_recharge', 'avg_call_duration', 'avg_data_usage',
            'churn_risk', 'complaints_last_month', 'support_tickets', 'last_interaction',
            'churn_reason'
        ]

        # Fill missing columns with default values
        for col in expected_columns:
            if col not in customers.columns:
                if col in ['monthly_charge', 'total_charges', 'avg_monthly_long_distance_charges',
                           'avg_monthly_gb_download', 'total_refunds', 'total_extra_data_charges',
                           'total_long_distance_charges', 'total_revenue', 'avg_call_duration',
                           'avg_data_usage', 'churn_risk']:
                    customers[col] = 0.0
                elif col in ['age', 'tenure_months', 'tenure', 'number_of_referrals',
                             'cltv', 'satisfaction_score', 'complaints_last_month', 'support_tickets']:
                    customers[col] = 0
                elif col in ['last_recharge', 'last_interaction']:
                    customers[col] = datetime.now()
                elif col == 'churn_reason':
                    customers[col] = "Unknown"
                else:
                    customers[col] = "Unknown"

        return customers

    except Exception as e:
        st.error(f"Error loading data from MongoDB: {str(e)}")
        return pd.DataFrame()

# Load model (replace with actual model loading)
@st.cache_resource
def load_churn_model():
    try:
        model = joblib.load('model1.pkl')
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

@st.cache_resource
def load_churn_scaler():
    try:
        model = joblib.load('num_scaler21.pkl')
        return model
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
scaler = load_churn_scaler()

def predict_churn(customer_data):
    """
    Predict churn risk for a customer using the trained model and scaler
    """
    try:
        # Load the model and scaler
        model = joblib.load('model1.pkl')
        scaler = joblib.load('num_scaler21.pkl')
        
        # Get the exact feature names the scaler was trained on
        if hasattr(scaler, 'feature_names_in_'):
            scaler_features = list(scaler.feature_names_in_)
        else:
            # Fallback if scaler doesn't have feature names
            scaler_features = [
                'Tenure Months', 'Monthly Charges', 'Total Charges', 'CLTV',
                'Number of Referrals', 'Avg Monthly Long Distance Charges',
                'Avg Monthly GB Download', 'Total Refunds', 
                'Total Extra Data Charges', 'Total Long Distance Charges',
                'Total Revenue', 'Satisfaction Score'
            ]
        
        # Get the exact feature names the model expects
        model_features = [
            'Zip Code', 'Gender', 'Senior Citizen', 'Partner', 'Dependents',
            'Tenure Months', 'Phone Service', 'Multiple Lines', 'Internet Service',
            'Online Security', 'Online Backup', 'Device Protection', 'Tech Support',
            'Streaming TV', 'Streaming Movies', 'Contract', 'Paperless Billing',
            'Payment Method', 'Monthly Charges', 'Total Charges', 'CLTV', 'Quarter',
            'Referred a Friend', 'Number of Referrals', 'Tenure in Months', 'Offer',
            'Phone Service_services', 'Avg Monthly Long Distance Charges',
            'Multiple Lines_services', 'Internet Service_services', 'Internet Type',
            'Avg Monthly GB Download', 'Online Security_services',
            'Online Backup_services', 'Device Protection Plan',
            'Premium Tech Support', 'Streaming TV_services',
            'Streaming Movies_services', 'Streaming Music', 'Unlimited Data',
            'Contract_services', 'Paperless Billing_services',
            'Payment Method_services', 'Monthly Charge', 'Total Charges_services',
            'Total Refunds', 'Total Extra Data Charges',
            'Total Long Distance Charges', 'Total Revenue', 'Quarter_status',
            'Satisfaction Score', 'CLTV_status'
        ]
        
        # Create DataFrame with all expected features in correct order
        input_df = pd.DataFrame(columns=model_features)
        
        # Fill in provided values, use safe defaults for missing ones
        for feature in model_features:
            if feature in customer_data:
                input_df[feature] = [customer_data[feature]]
            else:
                # Assign appropriate defaults based on feature type
                if feature in scaler_features or any(x in feature.lower() for x in ['charge', 'cltv', 'revenue', 'score']):
                    input_df[feature] = 0.0  # Numeric default
                else:
                    input_df[feature] = 'Missing'  # Categorical default
        
        # Scale only the numeric features that were in the training data
        numeric_cols = [col for col in scaler_features if col in input_df.columns]
        input_df[numeric_cols] = scaler.transform(input_df[numeric_cols])
        
        # Convert categoricals to numerical codes
        for col in input_df.select_dtypes(include=['object']).columns:
            input_df[col] = input_df[col].astype('category').cat.codes
            
        # Ensure final feature order matches model expectations
        input_df = input_df[model_features]
        
        # Convert to numpy array and predict
        input_np = input_df.values.astype(np.float32)
        prediction_proba = model.predict_proba(input_np)[:, 1]
        prediction = (prediction_proba >= 0.5).astype(int)
        
        return float(prediction_proba[0]), int(prediction[0])
        
    except Exception as e:
        st.error(f"Error making prediction: {e}")
        return None, None

def display_prediction_results(churn_prob, churn_pred, customer_data):
    if churn_prob is not None:
        st.markdown("---")
        st.subheader("Prediction Results")
        
        # Create columns for layout
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Gauge chart
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=churn_prob,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Churn Risk Score"},
                gauge={
                    'axis': {'range': [0, 1]},
                    'bar': {'color': "darkblue"},
                    'steps': [
                        {'range': [0, 0.3], 'color': "green"},
                        {'range': [0.3, 0.7], 'color': "orange"},
                        {'range': [0.7, 1], 'color': "red"}],
                    'threshold': {
                        'line': {'color': "black", 'width': 4},
                        'thickness': 0.75,
                        'value': churn_prob}
                }
            ))
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Prediction summary
            st.metric("Prediction", 
                     "Will Churn" if churn_pred == 1 else "Will Stay",
                     delta=f"{churn_prob:.1%} confidence")
            
            # Interpretation
            if churn_prob < 0.3:
                st.success("✅ Low churn risk")
                st.markdown("This customer is likely to stay with us.")
            elif churn_prob < 0.7:
                st.warning("⚠️ Medium churn risk")
                st.markdown("This customer may need attention.")
            else:
                st.error("🚨 High churn risk")
                st.markdown("Immediate action recommended!")
        
        # Display feature importance (if available)
        try:
            if hasattr(model, 'feature_importances_'):
                st.subheader("Top Influential Factors")
                feat_imp = pd.DataFrame({
                    'Feature': features,
                    'Importance': model.feature_importances_
                }).sort_values('Importance', ascending=False).head(10)
                
                fig = px.bar(feat_imp, x='Importance', y='Feature', 
                            orientation='h', title='Feature Importance')
                st.plotly_chart(fig, use_container_width=True)
        except Exception:
            pass
        
        # Save prediction button
        if st.button("Save Prediction to Database"):
            try:
                # Connect to MongoDB
                client = MongoClient(os.getenv("MONGO_URI"))
                db = client["customer_db"]
                collection = db["churn_predictions"]
                
                # Prepare document
                prediction_doc = {
                    "customer_id": customer_data.get('customer_id', 'CUST1001'),
                    "name": customer_data.get('name', 'Unknown'),
                    "prediction_date": datetime.now(),
                    "churn_probability": churn_prob,
                    "churn_prediction": bool(churn_pred),
                    "customer_data": customer_data
                }
                
                # Insert prediction
                collection.insert_one(prediction_doc)
                st.success("Prediction saved successfully!")
            except Exception as e:
                st.error(f"Error saving prediction: {e}")

def preprocess_customer_for_prediction(mongo_doc):
    yes_no_map = {"Yes": 1, "No": 0}
    gender_map = {"Male": 0, "Female": 1, "Other": 2}
    internet_service_map = {"DSL": 0, "Fiber optic": 1, "No": 2}
    multi_lines_map = {"No": 0, "Yes": 1, "No phone service": 2}
    internet_feature_map = {"No": 0, "Yes": 1, "No internet service": 2}
    contract_map = {"Month-to-month": 0, "One year": 1, "Two year": 2}
    payment_method_map = {
        "Electronic check": 0,
        "Mailed check": 1,
        "Bank transfer": 2,
        "Bank transfer (automatic)": 2,
        "Credit card": 3,
        "Credit card (automatic)": 3
    }
    offer_map = {"None": 0, "Offer A": 1, "Offer B": 2, "Offer C": 3, "Offer D": 4}
    internet_type_map = {"DSL": 0, "Fiber optic": 1, "None": 2}
    quarter_map = {"Q1": 1, "Q2": 2, "Q3": 3, "Q4": 4}
    quarter_status_map = {"Active": 1, "Inactive": 0}
    cltv_status_map = {"Low": 0, "Medium": 1, "High": 2}

    return {
        'Zip Code': str(mongo_doc.get('zip_code', '00000')),
        'Gender': gender_map.get(mongo_doc.get('gender', 'Male'), 0),
        'Senior Citizen': yes_no_map.get(mongo_doc.get('senior_citizen', 'No'), 0),
        'Partner': yes_no_map.get(mongo_doc.get('partner', 'No'), 0),
        'Dependents': yes_no_map.get(mongo_doc.get('dependents', 'No'), 0),
        'Tenure Months': int(mongo_doc.get('tenure_months', 0)),
        'Phone Service': yes_no_map.get(mongo_doc.get('phone_service', 'Yes'), 1),
        'Multiple Lines': multi_lines_map.get(mongo_doc.get('multiple_lines', 'No'), 0),
        'Internet Service': internet_service_map.get(mongo_doc.get('internet_service', 'DSL'), 0),
        'Online Security': internet_feature_map.get(mongo_doc.get('online_security', 'No'), 0),
        'Online Backup': internet_feature_map.get(mongo_doc.get('online_backup', 'No'), 0),
        'Device Protection': internet_feature_map.get(mongo_doc.get('device_protection', 'No'), 0),
        'Tech Support': internet_feature_map.get(mongo_doc.get('tech_support', 'No'), 0),
        'Streaming TV': internet_feature_map.get(mongo_doc.get('streaming_tv', 'No'), 0),
        'Streaming Movies': internet_feature_map.get(mongo_doc.get('streaming_movies', 'No'), 0),
        'Contract': contract_map.get(mongo_doc.get('contract', 'Month-to-month'), 0),
        'Paperless Billing': yes_no_map.get(mongo_doc.get('paperless_billing', 'Yes'), 1),
        'Payment Method': payment_method_map.get(mongo_doc.get('payment_method', 'Electronic check'), 0),
        'Monthly Charges': float(mongo_doc.get('monthly_charge', 0.0)),
        'Total Charges': float(mongo_doc.get('total_charges', 0.0)),
        'CLTV': int(mongo_doc.get('cltv', 0)),
        'Quarter': quarter_map.get(mongo_doc.get('quarter', 'Q1'), 1),
        'Referred a Friend': yes_no_map.get(mongo_doc.get('referred_a_friend', 'No'), 0),
        'Number of Referrals': int(mongo_doc.get('number_of_referrals', 0)),
        'Tenure in Months': int(mongo_doc.get('tenure_months', 0)),
        'Offer': offer_map.get(mongo_doc.get('offer', 'None'), 0),
        'Internet Type': internet_type_map.get(mongo_doc.get('internet_type', 'DSL'), 0),
        'Avg Monthly GB Download': float(mongo_doc.get('avg_monthly_gb_download', 0)),
        'Avg Monthly Long Distance Charges': float(mongo_doc.get('avg_monthly_long_distance_charges', 0)),
        'Streaming Music': yes_no_map.get(mongo_doc.get('streaming_music', 'No'), 0),
        'Unlimited Data': yes_no_map.get(mongo_doc.get('unlimited_data', 'No'), 0),
        'Total Revenue': float(mongo_doc.get('total_revenue', 0)),
        'Quarter_status': quarter_status_map.get(mongo_doc.get('cltv_status', 'Active'), 1),
        'Satisfaction Score': int(mongo_doc.get('satisfaction_score', 3)),
        'CLTV_status': cltv_status_map.get(mongo_doc.get('cltv_status_label', 'Medium'), 1)
    }

def ask_gemini(query, customers_df, filtered_df):
    prompt = f"""
    You are a data assistant. Answer the following question using the pandas DataFrames given.
    Two DataFrames are provided: `customer_df` (full dataset) and `filtered_df` (filtered view).

    If the question is about a specific customer, use the `customer_df` DataFrame.
    if for priority questions, use the `filtered_df` DataFrame and check tenure and churn_risk and plan to decide whom to priorotize.

    Respond in plain English.

    Question: {query}

    Do not hallucinate data. Use only what’s available in the DataFrames.
    Here are sample data:
    customer_df:
    {customers_df.to_markdown()}

    filtered_df:
    {filtered_df.head(10).to_markdown()}
    """
    try:
        response = gemini_model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"❌ Error from Gemini: {e}"

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
    
    st.markdown("""
    <style>
    .value-box {
        border-radius: 8px;
        padding: 20px;
        margin-bottom: 20px;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
    .value-box.high-risk {
        background: linear-gradient(135deg, #ff758c, #ff7eb3);
        color: white;
    }
    .value-box.total {
        background: linear-gradient(135deg, #76b852, #8DC26F);
        color: white;
    }
    .value-box h3 {
        margin-top: 0;
        font-size: 1.2rem;
    }
    .value-box .value {
        font-size: 2.5rem;
        font-weight: bold;
    }
    </style>
    """, unsafe_allow_html=True)

    st.markdown(f"""
    <div class="value-box high-risk">
        <h3>🚨 High Risk Customers</h3>
        <div class="value">{len(high_risk_df)}</div>
        <div>{len(high_risk_df)/len(filtered_df)*100:.1f}% of total</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f"""
    <div class="value-box total">
        <h3>👥 Total Customers</h3>
        <div class="value">{len(filtered_df)}</div>
    </div>
    """, unsafe_allow_html=True)
    
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
    st.markdown("""
    <style>
    [data-testid="stMetricLabel"] p {
        color: black !important;
    }
    [data-testid="stMetricValue"] {
        color: black !important;
    }
    </style>
    """, unsafe_allow_html=True)
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
    coll1, coll2 = st.columns(2)
    with coll1:
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
    with coll2:
        st.subheader("🤖 Ask Gemini About Churn Insights")

        query = st.text_input("Enter your question (e.g., 'Which plan type has the highest churn risk?')")

        if st.button("Ask Gemini"):
            with st.spinner("Thinking..."):
                answer = ask_gemini(query, customers_df, filtered_df)
                st.markdown("### Gemini's Response:")
                st.write(answer)

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
        st.markdown("""
        <style>
        .stTabs [data-baseweb="tab-list"] button {
            color: gray !important;
        }
        .stTabs [data-baseweb="tab-list"] button[aria-selected="true"] {
            color: red !important;
        }
        </style>
        """, unsafe_allow_html=True)
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
            st.markdown("""
            <style>
            div[data-testid="stForm"] .stSelectbox label {
                color: black !important;
            }
            div[data-testid="stForm"] .stTextArea label {
                color: black !important;
            }
            div[data-testid="stForm"] .stFormSubmitButton button {
                color: white !important;
            }
            div[data-testid="stForm"] .stFormSubmitButton button:hover {
                color: red !important;
            }
            </style>
            """, unsafe_allow_html=True)
            with st.form("action_plan"):
                selected_action = st.selectbox("Select action to implement", primary_recs)
                action_owner = st.selectbox("Assign to", ["Retention Team", "Account Manager", "Customer Support"])
                timeline = st.selectbox("Timeline", ["Immediate", "Within 24 hours", "Within 3 days"])
                notes = st.text_area("Additional notes")
                
                submitted = st.form_submit_button("Create Action Plan")
                if submitted:
                    st.success("Action plan created successfully!")
    
    else:
        st.info("Select a customer from the Dashboard to view detailed insights")

elif page == "Model Management":
    st.title("🤖 Model Management")
    
    st.markdown("""
    <style>
    .stTabs [data-baseweb="tab-list"] button {
        color: gray !important;
    }
    .stTabs [data-baseweb="tab-list"] button[aria-selected="true"] {
        color: red !important;
    }
    </style>
    """, unsafe_allow_html=True)
    tab1, tab2, tab3, tab4 = st.tabs(["Model Performance", "Retrain Model", "Data Quality", "Predict Churn"])
    
    with tab1:
        st.subheader("Current Model Performance")
        
        # Mock performance metrics
        st.markdown("""
        <style>
        [data-testid="stMetricLabel"] p {
            color: black !important;
        }
        [data-testid="stMetricValue"] {
            color: black !important;
        }
        </style>
        """, unsafe_allow_html=True)
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Accuracy", "0.95", "2% from last month")
        with col2:
            st.metric("Precision", "0.96", "-1% from last month")
        with col3:
            st.metric("Recall", "0.93", "3% from last month")
        with col4:
            st.metric("F1 Score", "0.94", "1% from last month")
        
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
        st.markdown("""
        <style>
        .stDateInput label {
            color: black !important;
        }
        </style>
        """, unsafe_allow_html=True)
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
        st.markdown("""
        <style>
        .stSelectbox label {
            color: black !important;
        }
        </style>
        """, unsafe_allow_html=True)
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
    with tab4:
        st.subheader("Customer Information")
        
        # Add a radio button to choose input method
        input_method = st.radio("Select Input Method:", 
                            ["Enter Manually", "Load from Database"],
                            horizontal=True)
        
        if input_method == "Load from Database":
            # Load sample data from MongoDB
            customers = load_sample_data()
            
            if not customers.empty:
                # Select customer from dropdown
                customer_options = customers['customer_id'] + " - " + customers['name']
                selected_customer = st.selectbox("Select Customer", customer_options)
                
                # Get the selected customer's data
                selected_customer_id = selected_customer.split(" - ")[0]
                customer_data = customers[customers['customer_id'] == selected_customer_id].iloc[0].to_dict()
                
                # Display the customer data (read-only)
                with st.expander("View Customer Data"):
                    st.json({k: v for k, v in customer_data.items() if not k.startswith('_')})
                
                # Make prediction button
                if st.button("Predict Churn Risk for Selected Customer"):
                    with st.spinner("Analyzing customer data..."):
                        cust_data = preprocess_customer_for_prediction(customer_data)
                        churn_prob, churn_pred = predict_churn(cust_data)
                        display_prediction_results(churn_prob, churn_pred, customer_data)
            else:
                st.warning("No customer data found in the database.")
        
        else:  # Manual input
            # Personal Information
            col1, col2, col3 = st.columns(3)
            with col1:
                customer_id = st.text_input("Customer ID", "CUST1001")
                name = st.text_input("Full Name", "John Doe")
                gender = st.selectbox("Gender", ["Male", "Female", "Other"])
                senior_citizen = st.selectbox("Senior Citizen", ["Yes", "No"])
            with col2:
                partner = st.selectbox("Partner", ["Yes", "No"])
                dependents = st.selectbox("Dependents", ["Yes", "No"])
                zip_code = st.text_input("Zip Code", "10001")
            with col3:
                tenure_months = st.number_input("Tenure (Months)", min_value=0, max_value=120, value=12)
                satisfaction_score = st.slider("Satisfaction Score (1-5)", 1, 5, 3)
            
            st.subheader("Service Details")
            
            # Service Information
            col1, col2, col3 = st.columns(3)
            with col1:
                phone_service = st.selectbox("Phone Service", ["Yes", "No"])
                multiple_lines = st.selectbox("Multiple Lines", ["No", "Yes", "No phone service"])
                internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
            with col2:
                online_security = st.selectbox("Online Security", ["No", "Yes", "No internet service"])
                online_backup = st.selectbox("Online Backup", ["No", "Yes", "No internet service"])
                device_protection = st.selectbox("Device Protection", ["No", "Yes", "No internet service"])
            with col3:
                tech_support = st.selectbox("Tech Support", ["No", "Yes", "No internet service"])
                streaming_tv = st.selectbox("Streaming TV", ["No", "Yes", "No internet service"])
                streaming_movies = st.selectbox("Streaming Movies", ["No", "Yes", "No internet service"])
            
            st.subheader("Contract & Billing")
            
            # Contract Information
            col1, col2, col3 = st.columns(3)
            with col1:
                contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
                paperless_billing = st.selectbox("Paperless Billing", ["Yes", "No"])
            with col2:
                payment_method = st.selectbox("Payment Method", [
                    "Electronic check", "Mailed check", 
                    "Bank transfer (automatic)", "Credit card (automatic)"
                ])
                referred_a_friend = st.selectbox("Referred a Friend", ["Yes", "No"])
            with col3:
                num_referrals = st.number_input("Number of Referrals", min_value=0, value=0)
                offer = st.selectbox("Current Offer", ["None", "Offer A", "Offer B", "Offer C", "Offer D"])
            
            st.subheader("Usage & Financial Metrics")
            
            # Usage and Financials
            col1, col2, col3 = st.columns(3)
            with col1:
                monthly_charges = st.number_input("Monthly Charges ($)", min_value=0.0, value=70.0)
                total_charges = st.number_input("Total Charges ($)", min_value=0.0, value=840.0)
            with col2:
                avg_monthly_gb = st.number_input("Avg Monthly GB Download", min_value=0.0, value=50.0)
                avg_monthly_long_dist = st.number_input("Avg Long Distance Charges ($)", min_value=0.0, value=10.0)
            with col3:
                cltv = st.number_input("Customer Lifetime Value", min_value=0, value=2000)
                total_revenue = st.number_input("Total Revenue ($)", min_value=0.0, value=1000.0)
            
            # Additional fields (collapsed by default)
            with st.expander("Advanced Features"):
                col1, col2, col3 = st.columns(3)
                with col1:
                    internet_type = st.selectbox("Internet Type", ["DSL", "Fiber optic", "None"])
                    unlimited_data = st.selectbox("Unlimited Data", ["Yes", "No"])
                with col2:
                    streaming_music = st.selectbox("Streaming Music", ["Yes", "No"])
                    quarter = st.selectbox("Quarter", ["Q1", "Q2", "Q3", "Q4"])
                with col3:
                    quarter_status = st.selectbox("Quarter Status", ["Active", "Inactive"])
                    cltv_status = st.selectbox("CLTV Status", ["High", "Medium", "Low"])
            
            submitted = st.button("Predict Churn Risk")
            
            if submitted:
                yes_no_map = {"Yes": 1, "No": 0}
                gender_map = {"Male": 0, "Female": 1, "Other": 2}
                internet_service_map = {"DSL": 0, "Fiber optic": 1, "No": 2}
                multi_lines_map = {"No": 0, "Yes": 1, "No phone service": 2}
                internet_feature_map = {"No": 0, "Yes": 1, "No internet service": 2}
                contract_map = {"Month-to-month": 0, "One year": 1, "Two year": 2}
                payment_method_map = {
                    "Electronic check": 0,
                    "Mailed check": 1,
                    "Bank transfer (automatic)": 2,
                    "Credit card (automatic)": 3
                }
                offer_map = {"None": 0, "Offer A": 1, "Offer B": 2, "Offer C": 3, "Offer D": 4}
                internet_type_map = {"DSL": 0, "Fiber optic": 1, "None": 2}
                quarter_map = {"Q1": 1, "Q2": 2, "Q3": 3, "Q4": 4}
                quarter_status_map = {"Active": 1, "Inactive": 0}
                cltv_status_map = {"Low": 0, "Medium": 1, "High": 2}

                # Prepare customer data with proper types
                customer_data = {
                    'Zip Code': str(zip_code),
                    'Gender': gender_map[gender],
                    'Senior Citizen': yes_no_map[senior_citizen],
                    'Partner': yes_no_map[partner],
                    'Dependents': yes_no_map[dependents],
                    'Tenure Months': int(tenure_months),
                    'Phone Service': yes_no_map[phone_service],
                    'Multiple Lines': multi_lines_map[multiple_lines],
                    'Internet Service': internet_service_map[internet_service],
                    'Online Security': internet_feature_map[online_security],
                    'Online Backup': internet_feature_map[online_backup],
                    'Device Protection': internet_feature_map[device_protection],
                    'Tech Support': internet_feature_map[tech_support],
                    'Streaming TV': internet_feature_map[streaming_tv],
                    'Streaming Movies': internet_feature_map[streaming_movies],
                    'Contract': contract_map[contract],
                    'Paperless Billing': yes_no_map[paperless_billing],
                    'Payment Method': payment_method_map[payment_method],
                    'Monthly Charges': float(monthly_charges),
                    'Total Charges': float(total_charges),
                    'CLTV': int(cltv),
                    'Quarter': quarter_map[quarter],
                    'Referred a Friend': yes_no_map[referred_a_friend],
                    'Number of Referrals': int(num_referrals),
                    'Tenure in Months': int(tenure_months),
                    'Offer': offer_map[offer],
                    'Internet Type': internet_type_map[internet_type],
                    'Avg Monthly GB Download': float(avg_monthly_gb),
                    'Avg Monthly Long Distance Charges': float(avg_monthly_long_dist),
                    'Streaming Music': yes_no_map[streaming_music],
                    'Unlimited Data': yes_no_map[unlimited_data],
                    'Total Revenue': float(total_revenue),
                    'Quarter_status': quarter_status_map[quarter_status],
                    'Satisfaction Score': int(satisfaction_score),
                    'CLTV_status': cltv_status_map[cltv_status]
                }
                
                with st.spinner("Analyzing customer data..."):
                    churn_prob, churn_pred = predict_churn(customer_data)
                    display_prediction_results(churn_prob, churn_pred, customer_data)

elif page == "Retention Actions":
    st.title("🛡️ Retention Actions")
    
    st.markdown("""
    <style>
    .stTabs [data-baseweb="tab-list"] button {
        color: gray !important;
    }
    .stTabs [data-baseweb="tab-list"] button[aria-selected="true"] {
        color: red !important;
    }
    </style>
    """, unsafe_allow_html=True)
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
            st.markdown("""
            <style>
            div[data-testid="stForm"] .stSelectbox label {
                color: black !important;
            }
            div[data-testid="stForm"] .stTextArea label {
                color: black !important;
            }
            div[data-testid="stForm"] .stTextInput label {
                color: black !important;
            }
            div[data-testid="stForm"] .stDateInput label {
                color: black !important;
            }
            div[data-testid="stForm"] .stNumberInput label {
                color: black !important;
            }
            div[data-testid="stForm"] .stFormSubmitButton button {
                color: white !important;
            }
            div[data-testid="stForm"] .stFormSubmitButton button:hover {
                color: red !important;
            }
            </style>
            """, unsafe_allow_html=True)
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
        st.markdown("""
        <style>
        .stMultiSelect label {
            color: black !important;
        }

        .stDateInput label {
            color: black !important;
        }
        </style>
        """, unsafe_allow_html=True)
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
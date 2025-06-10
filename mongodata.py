from pymongo import MongoClient
from dotenv import load_dotenv
import numpy as np
from datetime import datetime, timedelta
import os

load_dotenv()

# Connect to MongoDB (default local connection)
client = MongoClient(os.getenv("MONGO_URI"))

# Database and Collection
db = client["customer_db"]
collection = db["churn_data"]

# Customer data as dictionaries
customers = []

for i in range(70):
    tenure = int(np.random.randint(1, 72))
    internet = np.random.choice(['DSL', 'Fiber optic', 'No'])
    monthly = round(np.random.uniform(20, 120), 2)
    total = round(monthly * tenure + np.random.uniform(0, 50), 2)
    cltv = int(total + np.random.uniform(100, 1000))
    customers.append({
        "customer_id": f"CUST200{i+1}",
        "name": np.random.choice([
    "Emma Smith", "Liam Johnson", "Olivia Brown", "Noah Davis", "Ava Miller", 
    "Sophia Wilson", "Lucas Moore", "Mia Taylor", "Ethan Anderson", "Charlotte Thomas",
    "Amelia White", "James Harris", "Harper Lewis", "Benjamin Young", "Evelyn Hall",
    "Alexander Allen", "Abigail King", "Henry Scott", "Emily Wright", "Daniel Adams",
    "Elizabeth Baker", "Michael Nelson", "Ella Carter", "Sebastian Mitchell", "Grace Perez",
    "Jackson Roberts", "Chloe Turner", "Aiden Phillips", "Aria Campbell", "Matthew Parker",
    "Lily Evans", "David Edwards", "Scarlett Collins", "Joseph Stewart", "Zoey Morris",
    "Samuel Rogers", "Nora Cook", "Anthony Reed", "Hannah Morgan", "Andrew Bell",
    "Victoria Murphy", "Joshua Bailey", "Riley Rivera", "Leo Cooper", "Lillian Richardson",
    "Owen Cox", "Ellie Howard", "Dylan Ward", "Layla Hughes", "Gabriel Peterson",
    "Aubrey Gray", "Jack Ramirez", "Penelope James", "Nathan Watson", "Camila Brooks",
    "Caleb Kelly", "Savannah Sanders", "Julian Price", "Stella Bennett", "Isaac Wood",
    "Paisley Barnes", "Christian Ross", "Addison Henderson", "Aaron Coleman", "Aaliyah Jenkins",
    "Elijah Perry", "Lucy Powell", "Thomas Long", "Hazel Patterson", "Charles Russell",
    "Natalie Simmons", "Christopher Foster", "Brooklyn Bryant", "Jonathan Alexander", "Leah Butler"
]),
        "age": int(np.random.randint(18, 70)),
        "gender": np.random.choice(['Male', 'Female']),
        "senior_citizen": np.random.choice(['Yes', 'No']),
        "partner": np.random.choice(['Yes', 'No']),
        "dependents": np.random.choice(['Yes', 'No']),
        "tenure_months": tenure,
        "tenure": tenure,
        "phone_service": np.random.choice(['Yes', 'No']),
        "multiple_lines": np.random.choice(['Yes', 'No']),
        "internet_service": internet,
        "online_security": np.random.choice(['Yes', 'No']),
        "online_backup": np.random.choice(['Yes', 'No']),
        "device_protection": np.random.choice(['Yes', 'No']),
        "tech_support": np.random.choice(['Yes', 'No']),
        "streaming_tv": np.random.choice(['Yes', 'No']),
        "streaming_movies": np.random.choice(['Yes', 'No']),
        "streaming_music": np.random.choice(['Yes', 'No']),
        "unlimited_data": np.random.choice(['Yes', 'No']),
        "contract": np.random.choice(['Month-to-month', 'One year', 'Two year']),
        "paperless_billing": np.random.choice(['Yes', 'No']),
        "payment_method": np.random.choice(['Mailed check', 'Bank transfer', 'Credit card', 'Electronic check']),
        "monthly_charge": monthly,
        "total_charges": total,
        "cltv": cltv,
        "cltv_status": np.random.choice(['Active', 'Inactive']),
        "quarter": np.random.choice(['Q1', 'Q2', 'Q3', 'Q4']),
        "referred_a_friend": np.random.choice(['Yes', 'No']),
        "number_of_referrals": int(np.random.randint(0, 5)),
        "offer": np.random.choice(['Offer A', 'Offer B', 'Offer C', 'None']),
        "avg_monthly_long_distance_charges": round(np.random.uniform(0, 50), 2),
        "internet_type": internet,
        "avg_monthly_gb_download": int(np.random.randint(0, 100)),
        "total_refunds": round(np.random.uniform(0, 30), 2),
        "total_extra_data_charges": int(np.random.randint(0, 10)),
        "total_long_distance_charges": round(np.random.uniform(0, 2000), 2),
        "total_revenue": round(total + np.random.uniform(100, 500), 2),
        "satisfaction_score": int(np.random.randint(1, 6)),
        "location": np.random.choice(['Urban', 'Suburban', 'Rural']),
        "plan_type": np.random.choice(['Premium', 'Standard', 'Basic']),
        "last_recharge": datetime.now() - timedelta(days=int(np.random.randint(1, 60))),
        "avg_call_duration": round(np.random.uniform(5, 120), 2),
        "avg_data_usage": round(np.random.uniform(1, 20), 2),
        "churn_risk": round(np.random.uniform(0, 1), 2),
        "complaints_last_month": int(np.random.randint(0, 5)),
        "support_tickets": int(np.random.randint(0, 10)),
        "last_interaction": datetime.now() - timedelta(days=int(np.random.randint(1, 30))),
        "churn_reason": np.random.choice([
            "Price sensitivity", "Network issues", "Customer service", 
            "Competitor offer", "Relocation", "Dissatisfaction", 
            "Billing issues", "Data speed", "Coverage problems"
        ])
    })
# Insert into MongoDB
insert_result = collection.insert_many(customers)

# Print inserted IDs
print("Inserted IDs:", insert_result.inserted_ids)
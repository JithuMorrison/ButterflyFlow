
## 🦋 ButterflyFlow — AI-Driven Churn Defender for Telecoms

ButterflyFlow is a full-stack, **AI-powered churn-prevention platform** designed to help telecom customer-success teams retain high-value users.

It identifies early signals of potential churn, provides insights into *why* customers are disengaging, and empowers support teams with **automated, Gemini-generated retention strategies** — all through a sleek and intuitive **Streamlit dashboard**.

Whether it's billing dissatisfaction, poor network experience, or unresolved issues, ButterflyFlow not only **predicts who’s at risk** but also **tells you what to do about it** — including **one-click email execution** via Gmail API.


> 🏁 **Submitted to the Hackathon: Code the Cloud'25**

---
## ✨ Key Features

| # | Capability | What it Delivers |
|---|------------|------------------|
| **1. Predict‐Churn** | **Binary churn score** (0 – 1) using TabNet at 94 % accuracy. |
| **2. Churn Category** | Multi-class CatBoost model pinpoints *why* each user might churn (pricing, network, support wait-time, etc.). |
| **3. Interactive Dashboard** | Streamlit UI shows global KPIs, cohort heatmaps, and **per-customer 360° panels** (tenure, ARPU, usage, complaints). |
| **4. Risk Filtering** | Quick filters (e.g. **risk > 0.70**) surface high-priority customers so agents can act fast. |
| **5. Gemini Chatbot Assistant** | Embedded chat explains churn drivers in plain English and suggests personalized retention tactics. |
| **6. One-Click Email Outreach** | Agents approve Gemini’s script → Gmail API sends an on-brand email instantly. |
| **7. Campaign Mode** | Bulk-select churn segments and auto-launch retention campaigns, tracking open & response rates. |
| **8. Real-Time Updates** | New data streams into MongoDB; the dashboard refreshes without a full redeploy. |


---

## ⚙️ Tech Stack & Tools

| Category            | Technology                | Icons                                                                                                                                                                                                                                       |
| ------------------- | ------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 🧠 Machine Learning | TabNet (Google), CatBoost | ![Python](https://img.shields.io/badge/Python-3776AB?logo=python\&logoColor=white) ![CatBoost](https://img.shields.io/badge/CatBoost-yellow?logo=python\&logoColor=black) ![Google](https://img.shields.io/badge/TabNet-Google-brightgreen) |
| 🧰 Backend/Storage  | MongoDB Atlas             | ![MongoDB](https://img.shields.io/badge/MongoDB-47A248?logo=mongodb\&logoColor=white)                                                                                                                                                       |
| 🖼️ Frontend        | Streamlit                 | ![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?logo=streamlit\&logoColor=white)                                                                                                                                                 |
| 🔮 AI Assistant     | Gemini API                | ![Gemini](https://img.shields.io/badge/Gemini-API-blueviolet?logo=google\&logoColor=white)                                                                                                                                                  |
| 📧 Email Automation | Gmail API                 | ![Gmail](https://img.shields.io/badge/Gmail-API-EA4335?logo=gmail\&logoColor=white)                                                                                                                                                         |
| 📈 Deployment-Ready | Python 3.10+, virtualenv  | ![Python](https://img.shields.io/badge/Runtime-Python%203.10+-blue)                                                                                                                                                                         |

---

## 📊 Dataset

- **Source:** [Kaggle – Telco Customer Churn 11-13](https://www.kaggle.com/datasets/ylchang/telco-customer-churn-1113)  
- **Rows:** 7 043 customers × 21 raw features  
- **Pipeline:** outlier clipping · categorical encoding · 30-day rolling aggregates · class balancing.

---
## 🖼️ Screenshots

<img src="https://github.com/user-attachments/assets/dc76f05d-31e2-4332-8a95-55e7960fbba7" width="500"/>
<img src="https://github.com/user-attachments/assets/c1c0e6ee-5429-4558-b397-2445d160a3ad" width="500"/>
<img src="https://github.com/user-attachments/assets/7df600e5-5969-4d65-9004-6b978d5d1d2a" width="500"/>
<img src="https://github.com/user-attachments/assets/1368ce85-f828-464c-a838-c18eb4a99121" width="500"/>

<img src="https://github.com/user-attachments/assets/4bc32958-16c6-455e-95b8-61f8e2669dbd" width="500"/>

<img src="https://github.com/user-attachments/assets/03e34b54-bb99-4ebd-a9da-f3fdd20f8beb" width="500"/>

<img src="https://github.com/user-attachments/assets/1760efb9-abe5-402b-b893-862b797a0de1" width="500"/>
<img src="https://github.com/user-attachments/assets/cf05c570-a530-4ebb-b16f-1c4218ba20a5" width="500"/>
<img src="https://github.com/user-attachments/assets/d7896e33-3c1a-499e-bad4-dacb1685417b" width="500"/>

---
## 🚀 Getting Started

### 🔧 Setup

1. **Clone this repo:**

   ```bash
   git clone https://github.com/your-username/butterflyflow.git
   cd butterflyflow
   ```

2. **Create and activate a virtual environment:**

   ```bash
   python -m venv .venv
   .venv\Scripts\activate      # For Windows
   source .venv/bin/activate   # For macOS/Linux
   ```

3. **Install dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

4. **Set your environment variables** (either via `.env` or manually):

   ```bash
   MONGODB_URI="your-mongodb-connection-string"
   GEMINI_API_KEY="your-gemini-key"
   GOOGLE_CREDENTIALS_JSON="path/to/gmail_service_account.json"
   ```

---

### ▶️ Run the Application

To launch the dashboard using Streamlit:

```bash
python -m streamlit run "C:/link/AIChurn/AIChurn.py"
```

Once launched, open your browser and navigate to:

```
http://localhost:8501
```

You're now ready to explore churn predictions, insights, and retention strategies in real-time! 🎯

---
MIT License

Copyright (c) 2025 Kowshika

Permission is hereby granted, free of charge, to any person obtaining a copy...

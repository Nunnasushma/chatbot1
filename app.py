import os
import pandas as pd
import numpy as np
import streamlit as st
import joblib

# 1. Page Configuration
st.set_page_config(
    page_title="Car Price Predictor",
    page_icon="🚗",
    layout="centered",
    initial_sidebar_state="expanded"
)

# 2. Inject Custom Premium Styling CSS
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');

/* Global Reset & Typography */
html, body, [class*="css"] {
    font-family: 'Plus Jakarta Sans', sans-serif !important;
}

/* Background Theme */
.stApp {
    background: radial-gradient(circle at 50% 0%, #1e1b4b 0%, #0f172a 60%, #020617 100%);
    color: #f8fafc;
}

/* Custom Header Banner */
.header-container {
    text-align: center;
    padding: 2.2rem 1.5rem 1.8rem 1.5rem;
    margin-bottom: 2rem;
    background: rgba(30, 41, 59, 0.45);
    backdrop-filter: blur(16px);
    -webkit-backdrop-filter: blur(16px);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 24px;
    box-shadow: 0 20px 40px -15px rgba(0, 0, 0, 0.5);
}

.header-badge {
    display: inline-flex;
    align-items: center;
    gap: 0.5rem;
    padding: 0.4rem 1.2rem;
    background: linear-gradient(90deg, rgba(99, 102, 241, 0.25), rgba(168, 85, 247, 0.25));
    border: 1px solid rgba(168, 85, 247, 0.4);
    border-radius: 9999px;
    color: #c084fc;
    font-size: 0.85rem;
    font-weight: 700;
    letter-spacing: 0.06em;
    margin-bottom: 1rem;
    text-transform: uppercase;
}

.header-title {
    font-size: 2.6rem;
    font-weight: 800;
    background: linear-gradient(135deg, #ffffff 0%, #e2e8f0 50%, #818cf8 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0.6rem;
    line-height: 1.2;
}

.header-subtitle {
    font-size: 1.05rem;
    color: #94a3b8;
    max-width: 580px;
    margin: 0 auto;
    line-height: 1.5;
}

/* Glassmorphism Card Form */
div[data-testid="stForm"] {
    background: rgba(15, 23, 42, 0.65) !important;
    backdrop-filter: blur(16px) !important;
    -webkit-backdrop-filter: blur(16px) !important;
    border: 1px solid rgba(255, 255, 255, 0.1) !important;
    border-radius: 24px !important;
    padding: 2.2rem !important;
    box-shadow: 0 25px 50px -12px rgba(0, 0, 0, 0.6) !important;
}

/* Form Section Title */
.form-section-header {
    font-size: 1.25rem;
    font-weight: 700;
    color: #f1f5f9;
    margin-bottom: 1.2rem;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}

/* Input Labels & Selectboxes */
.stSelectbox label, .stNumberInput label {
    font-weight: 600 !important;
    color: #cbd5e1 !important;
    font-size: 0.92rem !important;
}

div[data-baseweb="select"] > div, div[data-baseweb="input"] > div {
    background-color: rgba(30, 41, 59, 0.85) !important;
    border: 1px solid rgba(255, 255, 255, 0.12) !important;
    border-radius: 12px !important;
    color: #ffffff !important;
    transition: all 0.2s ease-in-out;
}

div[data-baseweb="select"] > div:hover, div[data-baseweb="input"] > div:hover {
    border-color: #818cf8 !important;
    box-shadow: 0 0 12px rgba(99, 102, 241, 0.25) !important;
}

/* Form Submit Button */
div[data-testid="stFormSubmitButton"] {
    margin-top: 1.5rem !important;
}

div[data-testid="stFormSubmitButton"] > button {
    width: 100% !important;
    background: linear-gradient(90deg, #6366f1 0%, #8b5cf6 50%, #d946ef 100%) !important;
    color: #ffffff !important;
    font-weight: 700 !important;
    font-size: 1.05rem !important;
    padding: 0.85rem 2rem !important;
    border-radius: 14px !important;
    border: none !important;
    box-shadow: 0 10px 30px -5px rgba(99, 102, 241, 0.5) !important;
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
    cursor: pointer !important;
    text-transform: uppercase !important;
    letter-spacing: 0.06em !important;
}

div[data-testid="stFormSubmitButton"] > button:hover {
    transform: translateY(-2px) scale(1.01) !important;
    box-shadow: 0 15px 35px -5px rgba(168, 85, 247, 0.6) !important;
}

/* Result Card */
.result-card {
    background: radial-gradient(circle at 50% 0%, rgba(99, 102, 241, 0.25) 0%, rgba(15, 23, 42, 0.95) 100%);
    border: 1px solid rgba(168, 85, 247, 0.4);
    border-radius: 24px;
    padding: 2.2rem 1.8rem;
    text-align: center;
    margin-top: 2rem;
    box-shadow: 0 20px 50px -10px rgba(0, 0, 0, 0.7);
    animation: fadeIn 0.5s ease-out;
}

.result-badge {
    display: inline-block;
    padding: 0.3rem 0.9rem;
    background: rgba(168, 85, 247, 0.2);
    border: 1px solid rgba(168, 85, 247, 0.4);
    border-radius: 9999px;
    color: #e9d5ff;
    font-size: 0.8rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-bottom: 0.75rem;
}

.price-display-lakhs {
    font-size: 3.6rem;
    font-weight: 800;
    background: linear-gradient(135deg, #38bdf8 0%, #818cf8 50%, #c084fc 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    line-height: 1.1;
    margin: 0.3rem 0;
}

.price-display-total {
    color: #94a3b8;
    font-size: 1.1rem;
    font-weight: 600;
    margin-bottom: 1.5rem;
}

/* Specifications Grid */
.spec-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(130px, 1fr));
    gap: 0.75rem;
    margin-top: 1.5rem;
}

.spec-badge {
    background: rgba(30, 41, 59, 0.75);
    border: 1px solid rgba(255, 255, 255, 0.1);
    border-radius: 14px;
    padding: 0.75rem 0.6rem;
    text-align: center;
}

.spec-label {
    font-size: 0.72rem;
    color: #94a3b8;
    text-transform: uppercase;
    font-weight: 600;
    letter-spacing: 0.05em;
    margin-bottom: 0.2rem;
}

.spec-value {
    font-size: 0.95rem;
    color: #f8fafc;
    font-weight: 700;
}

/* Sidebar styling */
section[data-testid="stSidebar"] {
    background: rgba(15, 23, 42, 0.95) !important;
    border-right: 1px solid rgba(255, 255, 255, 0.08) !important;
}

.stat-card {
    background: rgba(30, 41, 59, 0.5);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 14px;
    padding: 1rem;
    margin-bottom: 0.8rem;
}

.stat-title {
    font-size: 0.78rem;
    color: #94a3b8;
    text-transform: uppercase;
    font-weight: 600;
}

.stat-value {
    font-size: 1.25rem;
    color: #38bdf8;
    font-weight: 700;
}

@keyframes fadeIn {
    from { opacity: 0; transform: translateY(12px); }
    to { opacity: 1; transform: translateY(0); }
}
</style>
""", unsafe_allow_html=True)

# 3. Model & Data Paths
MODEL_PATH = os.path.join("model", "car_price_model.pkl")
DATASET_PATH = os.path.join("dataset", "car_data.csv")

# 4. Verify Model Existence
if not os.path.exists(MODEL_PATH):
    st.error(
        "⚠️ **Model file not found!** (`model/car_price_model.pkl`)\n\n"
        "Please train the machine learning model first by running `python train_model.py` in your terminal."
    )
    st.stop()

# 5. Load Trained Pipeline
@st.cache_resource
def load_trained_pipeline():
    return joblib.load(MODEL_PATH)

try:
    pipeline = load_trained_pipeline()
except Exception as e:
    st.error(f"❌ Error loading trained model pipeline: {e}")
    st.stop()

# 6. Load Dataset Models
@st.cache_data
def get_car_models():
    if os.path.exists(DATASET_PATH):
        try:
            df = pd.read_csv(DATASET_PATH)
            if 'name' in df.columns:
                return sorted(df['name'].dropna().unique().tolist())
        except Exception:
            pass
    return [
        "Honda City", "Maruti Swift Dzire VDI", "Maruti Swift VXI BSIII",
        "Hyundai i20 Sportz Diesel", "Skoda Rapid 1.5 TDI Ambition",
        "Toyota Etios VXD", "Ford Figo Diesel", "Renault Duster"
    ]

car_models_list = get_car_models()

# 7. Sidebar Content
with st.sidebar:
    st.markdown("### 📊 Model Metrics")
    
    st.markdown("""<div class="stat-card">
<div class="stat-title">Algorithm</div>
<div class="stat-value" style="color: #c084fc;">Random Forest</div>
</div>
<div class="stat-card">
<div class="stat-title">R² Accuracy Score</div>
<div class="stat-value">72.4%</div>
</div>
<div class="stat-card">
<div class="stat-title">Mean Absolute Error</div>
<div class="stat-value" style="color: #34d399;">₹1.75 Lakhs</div>
</div>""", unsafe_allow_html=True)
    
    st.markdown("---")
    st.markdown("### ℹ️ About")
    st.caption(
        "This application uses a Scikit-Learn Random Forest Regressor trained on Kaggle used car listing data. "
        "Categorical features are preprocessed via `OneHotEncoder` within a unified Scikit-Learn `Pipeline`."
    )

# 8. Main App Header Banner
st.markdown("""<div class="header-container">
<div class="header-badge">⚡ AI-Powered Valuation</div>
<div class="header-title">Car Price Predictor</div>
<div class="header-subtitle">
Estimate the expected market value of any vehicle based on model history, age, mileage, and specifications.
</div>
</div>""", unsafe_allow_html=True)

# 9. Main Input Form
with st.form("car_price_form"):
    st.markdown('<div class="form-section-header">🚘 Vehicle Specifications</div>', unsafe_allow_html=True)
    
    # 1. Car Model Selection
    selected_car_model = st.selectbox(
        "Car Model / Variant",
        options=car_models_list,
        index=car_models_list.index("Honda City") if "Honda City" in car_models_list else 0,
        help="Select the brand & model of the car."
    )

    col1, col2 = st.columns(2)

    with col1:
        # 2. Manufacturing Year
        year = st.number_input(
            "Manufacturing Year",
            min_value=1990,
            max_value=2026,
            value=2020,
            step=1,
            help="Year the car was manufactured."
        )

        # 3. Kilometers Driven
        km_driven = st.number_input(
            "Kilometers Driven (KMs)",
            min_value=0,
            max_value=1000000,
            value=30000,
            step=1000,
            help="Total distance driven."
        )

        # 4. Fuel Type
        fuel = st.selectbox(
            "Fuel Type",
            options=["Petrol", "Diesel", "CNG", "LPG"],
            index=0,
            help="Fuel variant."
        )

    with col2:
        # 5. Seller Type
        seller_type = st.selectbox(
            "Seller Type",
            options=["Individual", "Dealer", "Trustmark Dealer"],
            index=0,
            help="Type of seller."
        )

        # 6. Transmission
        transmission = st.selectbox(
            "Transmission",
            options=["Manual", "Automatic"],
            index=0,
            help="Gearbox type."
        )

        # 7. Previous Owners
        num_owners = st.number_input(
            "Previous Owners",
            min_value=0,
            max_value=10,
            value=1,
            step=1,
            help="Number of previous owners."
        )

        owner_mapping = {
            0: "Test Drive Car",
            1: "First Owner",
            2: "Second Owner",
            3: "Third Owner"
        }
        owner_str = owner_mapping.get(int(num_owners), "Fourth & Above Owner")

    submit_btn = st.form_submit_button("✨ CALCULATE ESTIMATED PRICE")

# 10. Prediction Processing & Output Display
if submit_btn:
    if km_driven < 0:
        st.error("❌ Kilometers driven cannot be negative.")
    elif year < 1990 or year > 2026:
        st.error("❌ Please enter a valid manufacturing year between 1990 and 2026.")
    else:
        input_data = pd.DataFrame([{
            'name': selected_car_model,
            'year': int(year),
            'km_driven': float(km_driven),
            'fuel': fuel,
            'seller_type': seller_type,
            'transmission': transmission,
            'owner': owner_str
        }])

        try:
            predicted_price_inr = float(pipeline.predict(input_data)[0])
            predicted_price_inr = max(0.0, predicted_price_inr)
            price_in_lakhs = predicted_price_inr / 100000.0

            # Render Premium Result Card
            st.markdown(f"""<div class="result-card">
<div class="result-badge">💰 Estimated Market Value</div>
<div class="price-display-lakhs">₹{price_in_lakhs:.2f} Lakhs</div>
<div class="price-display-total">₹{predicted_price_inr:,.0f} INR</div>
</div>""", unsafe_allow_html=True)

        except Exception as err:
            st.error(f"❌ Prediction failed: {err}")

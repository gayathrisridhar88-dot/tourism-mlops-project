import streamlit as st
import pandas as pd
from huggingface_hub import hf_hub_download
import joblib

# Load model from Hugging Face
model_path = hf_hub_download(
    repo_id="gayathri1909/tourism-model",
    filename="best_model.pkl"
)
model = joblib.load(model_path)

# App Title
st.title("Tourism Package Purchase Prediction")
st.write("""
This application predicts whether a customer is likely to purchase the Wellness Tourism Package.
Please enter customer details below.
""")

# =========================
# User Inputs
# =========================

Age = st.number_input("Age", min_value=18, max_value=100, value=30)
TypeofContact = st.selectbox("Type of Contact", ["Company Invited", "Self Inquiry"])
CityTier = st.selectbox("City Tier", [1, 2, 3])
DurationOfPitch = st.number_input("Duration of Pitch", min_value=0.0, value=10.0)
Occupation = st.selectbox("Occupation", ["Salaried", "Small Business", "Free Lancer"])
Gender = st.selectbox("Gender", ["Male", "Female"])
NumberOfPersonVisiting = st.number_input("Number of Persons Visiting", min_value=1, value=2)
NumberOfFollowups = st.number_input("Number of Followups", min_value=0, value=2)
ProductPitched = st.selectbox("Product Pitched", ["Basic", "Standard", "Deluxe", "Super Deluxe", "King"])
PreferredPropertyStar = st.selectbox("Preferred Property Star", [1, 2, 3, 4, 5])
MaritalStatus = st.selectbox("Marital Status", ["Single", "Married", "Divorced"])
NumberOfTrips = st.number_input("Number of Trips", min_value=0, value=2)
Passport = st.selectbox("Passport", [0, 1])
PitchSatisfactionScore = st.selectbox("Pitch Satisfaction Score", [1, 2, 3, 4, 5])
OwnCar = st.selectbox("Own Car", [0, 1])
NumberOfChildrenVisiting = st.number_input("Number of Children Visiting", min_value=0, value=0)
Designation = st.selectbox("Designation", ["Manager", "Executive", "Senior Manager", "AVP", "VP"])
MonthlyIncome = st.number_input("Monthly Income", min_value=0, value=30000)

# =========================
# Create Input DataFrame
# =========================

input_data = pd.DataFrame([{
    'Age': Age,
    'TypeofContact': TypeofContact,
    'CityTier': CityTier,
    'DurationOfPitch': DurationOfPitch,
    'Occupation': Occupation,
    'Gender': Gender,
    'NumberOfPersonVisiting': NumberOfPersonVisiting,
    'NumberOfFollowups': NumberOfFollowups,
    'ProductPitched': ProductPitched,
    'PreferredPropertyStar': PreferredPropertyStar,
    'MaritalStatus': MaritalStatus,
    'NumberOfTrips': NumberOfTrips,
    'Passport': Passport,
    'PitchSatisfactionScore': PitchSatisfactionScore,
    'OwnCar': OwnCar,
    'NumberOfChildrenVisiting': NumberOfChildrenVisiting,
    'Designation': Designation,
    'MonthlyIncome': MonthlyIncome
}])

# =========================
# Prediction
# =========================

if st.button("Predict"):
    prediction = model.predict(input_data)[0]

    result = "Customer WILL Purchase Package" if prediction == 1 else "Customer WILL NOT Purchase Package"

    st.subheader("Prediction Result:")
    st.success(result)

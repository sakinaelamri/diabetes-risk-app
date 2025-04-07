
import streamlit as st
import numpy as np
import pickle

# Load the model
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

# Title
st.title("Early Diabetes Risk Detector")

st.markdown("This app uses machine learning to predict the risk of diabetes **without relying on glucose levels**. "
            "It was built by Sakina El-Amri using engineered health features and a tuned XGBoost model.")

# Input fields
age = st.number_input("Age", min_value=10, max_value=100, value=30)
bmi = st.number_input("BMI (Body Mass Index)", min_value=10.0, max_value=60.0, value=22.0)
insulin = st.number_input("Insulin Level (mu U/ml)", min_value=1.0, max_value=500.0, value=85.0)
bp = st.number_input("Blood Pressure (mm Hg)", min_value=40.0, max_value=180.0, value=70.0)

skin_option = st.radio("Do you know your skin thickness?", ["Yes", "No (Use default)"])
if skin_option == "Yes":
    skinfold = st.number_input("Skin Thickness (mm)", min_value=1.0, max_value=100.0, value=20.0)
else:
    skinfold = 20.0

pregnancies = st.number_input("Pregnancies", min_value=0, max_value=20, value=2)

dpf_option = st.selectbox(
    "Family History of Diabetes (Diabetes Pedigree Function)",
    [
        "No family history (0.2)",
        "One parent/sibling (0.6)",
        "Multiple close relatives (1.0)",
        "Very strong family history (1.5)"
    ]
)

dpf_map = {
    "No family history (0.2)": 0.2,
    "One parent/sibling (0.6)": 0.6,
    "Multiple close relatives (1.0)": 1.0,
    "Very strong family history (1.5)": 1.5
}

pedigree = dpf_map[dpf_option]

# Feature engineering
ins_gluc_ratio = insulin / 100
bmi_age_ratio = bmi / age
bp_skin = bp * skinfold
metabolic_stress = bmi * ins_gluc_ratio
ins_age = insulin * age
complex_stress = (bmi * insulin) / (skinfold + 1)

if st.button("Check Risk"):
    input_data = np.array([[pregnancies, bp, skinfold, insulin, bmi, pedigree, age,
                            ins_gluc_ratio, bmi_age_ratio, bp_skin, metabolic_stress, ins_age, complex_stress]])
    prediction = model.predict(input_data)[0]
    if prediction == 1:
        st.error("⚠️ High Risk: This profile may indicate early metabolic drift toward diabetes.")
    else:
        st.success("✅ Low Risk: No strong signs of early diabetes based on this profile.")

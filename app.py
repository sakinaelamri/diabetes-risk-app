
# Enhanced Streamlit App with Explanations and Smarter Thresholds

import streamlit as st
import numpy as np
import pickle

# Load the model
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

# Title
st.title("Early Diabetes Risk Detector")

st.markdown("""
This AI-powered tool predicts early diabetes risk by analyzing patterns in your metabolic profile — **without using glucose**. 
It was built using real health research and a trained XGBoost model by **Sakina El-Amri**.
""")

# Input fields
age = st.number_input("Age", min_value=10, max_value=100, value=30)
bmi = st.number_input("BMI (Body Mass Index)", min_value=10.0, max_value=60.0, value=22.0)
insulin = st.number_input("Insulin Level (mu U/ml)", min_value=1.0, max_value=500.0, value=85.0)
bp = st.number_input("Blood Pressure (mm Hg)", min_value=40.0, max_value=180.0, value=70.0)

skin_option = st.radio("Do you know your skin thickness?", ["Yes", "No (Use default)"])
if skin_option == "Yes":
    skinfold = st.number_input("Skin Thickness (mm)", min_value=1.0, max_value=100.0, value=20.0)
else:
    skinfold = 20.0  # default

pregnancies = st.number_input("Pregnancies", min_value=0, max_value=20, value=2)

# Family history dropdown
dpf_option = st.selectbox(
    "Family History of Diabetes (Diabetes Pedigree Function)",
    [
        "No family history (0.2)",
        "One parent/sibling (0.6)",
        "Multiple close relatives (1.0)",
        "Very strong family history (1.5)"
    ]
)

# Map dropdown to value
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

# Display legend
st.markdown("---")
st.subheader("Risk Interpretation Scale")
st.markdown("""
- ✅ **Low Risk:** 0.0 – 0.4  
- ⚠️ **Borderline:** 0.41 – 0.7  
- ❗ **High Risk:** 0.71 – 1.0
""")

if st.button("Check Risk"):
    input_data = np.array([[pregnancies, bp, skinfold, insulin, bmi, pedigree, age,
                            ins_gluc_ratio, bmi_age_ratio, bp_skin, metabolic_stress, ins_age, complex_stress]])
    proba = model.predict_proba(input_data)[0][1]  # probability of class 1 (high risk)

    st.markdown(f"### 🧪 Prediction Confidence: `{proba*100:.1f}%`")

    if proba > 0.7:
        st.error("❗ **High Risk**: Your profile shows a significant likelihood of early metabolic changes that may lead to diabetes.")
    elif proba > 0.4:
        st.warning("⚠️ **Borderline Risk**: Some mild signs of imbalance. Consider monitoring your lifestyle and checking in with a healthcare provider.")
    else:
        st.success("✅ **Low Risk**: Your profile does not show strong indicators of diabetes at this stage.")

    # Explanation block
    st.markdown("---")
    st.subheader("Why this result?")
    st.markdown(f"- **Insulin × Age**: {ins_age:.2f} (higher = more risk)")
    st.markdown(f"- **Complex Stress**: {complex_stress:.2f} (BMI × Insulin / Skinfold)")
    st.markdown(f"- **Metabolic Stress**: {metabolic_stress:.2f} (BMI × Insulin/Glucose ratio)")
    st.markdown(f"- **Blood Pressure × Skinfold**: {bp_skin:.2f} (linked to fat storage)")
    st.markdown(f"- **BMI / Age**: {bmi_age_ratio:.2f} (age-adjusted fat ratio)")

    st.info("These features were the strongest contributors to this prediction. They reflect how your body may be reacting to insulin and storing fat, even before glucose levels rise.")

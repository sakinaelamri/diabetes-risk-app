
import streamlit as st
import numpy as np
import pickle

# Load model
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

st.title("Early Diabetes Risk Detector")

st.markdown("""
This AI tool estimates early diabetes risk by analyzing insulin sensitivity and fat storage — without requiring a lab report.
""")

# Inputs
age = st.number_input("Age", min_value=10, max_value=100, value=30)
bmi = st.number_input("BMI", min_value=10.0, max_value=60.0, value=22.0)
insulin = st.number_input("Insulin (μU/mL)", min_value=1.0, max_value=500.0, value=85.0)
bp = st.number_input("Blood Pressure (mmHg)", min_value=40.0, max_value=180.0, value=70.0)
skin_option = st.radio("Do you know your skin thickness?", ["Yes", "No (Use default)"])
if skin_option == "Yes":
    skinfold = st.number_input("Skin Thickness (mm)", min_value=1.0, max_value=100.0, value=20.0)
else:
    skinfold = 20.0

pregnancies = st.number_input("Pregnancies", min_value=0, max_value=20, value=2)
glucose_range = st.selectbox("Approximate Glucose Level (optional)", [
    "I don't know (use 100)", "Normal (80–100)", "Pre-diabetic (101–125)", "Diabetic (126+)"
])
glucose_map = {
    "I don't know (use 100)": 100,
    "Normal (80–100)": 90,
    "Pre-diabetic (101–125)": 110,
    "Diabetic (126+)": 140
}
glucose = glucose_map[glucose_range]

dpf_option = st.selectbox("Family History of Diabetes", [
    "No family history (0.2)", "One parent/sibling (0.6)", "Multiple close relatives (1.0)", "Very strong family history (1.5)"
])
dpf_map = {
    "No family history (0.2)": 0.2,
    "One parent/sibling (0.6)": 0.6,
    "Multiple close relatives (1.0)": 1.0,
    "Very strong family history (1.5)": 1.5
}
pedigree = dpf_map[dpf_option]

# Feature engineering with caps
ins_gluc_ratio = insulin / glucose
bmi_age_ratio = bmi / age
bp_skin = min(bp * skinfold, 3000)
metabolic_stress = min(bmi * ins_gluc_ratio, )
ins_age = min(insulin * age, 30)
complex_stress = min((bmi * insulin) / (skinfold + 1), 3000)

st.markdown("---")
if st.button("Estimate My Risk"):
    input_data = np.array([[pregnancies, bp, skinfold, insulin, bmi, pedigree, age,
                            ins_gluc_ratio, bmi_age_ratio, bp_skin, metabolic_stress, ins_age, complex_stress]])
    proba = model.predict_proba(input_data)[0][1]

    st.markdown(f"### 🧪 Estimated Risk Score: **{proba*100:.1f}%**")

    if proba < 0.7:
        st.success("Low risk – no immediate concern.")
    elif proba < 0.9:
        st.warning("Borderline – something to monitor over time.")
    else:
        st.info("High signal – worth professional discussion.")

    st.markdown("---")
    st.subheader("Why this result?")
    st.markdown(f"- **Insulin × Age** (capped): {ins_age:.2f}")
    st.markdown(f"- **Complex Stress** (capped): {complex_stress:.2f}")
    st.markdown(f"- **Metabolic Stress** (capped): {metabolic_stress:.2f}")
    st.markdown(f"- **BP × Skinfold** (capped): {bp_skin:.2f}")
    st.markdown(f"- **BMI / Age**: {bmi_age_ratio:.2f}")
    st.caption("These values are interpreted within safe bounds to ensure balanced predictions.")

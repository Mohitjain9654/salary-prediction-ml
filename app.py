import streamlit as st
import pandas as pd
import joblib

# Load the trained model
model = joblib.load("salary_prediction_model.joblib")

st.set_page_config(page_title="Employee Salary Prediction", page_icon="💼", layout="centered")
st.title("💼 Employee Salary Prediction App")
st.markdown("Predict whether an employee earns >50K or ≤50K based on input features.")

st.sidebar.header("Input Employee Details")

# Sidebar Inputs
age = st.sidebar.slider("Age", 18, 70, 30)
workclass = st.sidebar.selectbox("Workclass", [
    "Private", "Self-emp-not-inc", "Self-emp-inc", "Federal-gov",
    "Local-gov", "State-gov", "Without-pay", "Never-worked"
])
education = st.sidebar.selectbox("Education", [
    "Bachelors", "HS-grad", "Some-college", "Masters", "Assoc-acdm",
    "Assoc-voc", "11th", "10th", "7th-8th", "Prof-school", "1st-4th"
])
educational_num = st.sidebar.slider("Education Number", 1, 16, 13)
occupation = st.sidebar.selectbox("Occupation", [
    "Tech-support", "Craft-repair", "Other-service", "Sales", "Exec-managerial",
    "Prof-specialty", "Handlers-cleaners", "Machine-op-inspct", "Adm-clerical",
    "Farming-fishing", "Transport-moving", "Priv-house-serv", "Protective-serv", "Armed-Forces"
])
relationship = st.sidebar.selectbox("Relationship", [
    "Husband", "Wife", "Own-child", "Not-in-family", "Other-relative", "Unmarried"
])
marital_status = st.sidebar.selectbox("Marital Status", [
    "Married-civ-spouse", "Never-married", "Divorced", "Separated", "Widowed", "Married-spouse-absent"
])
race = st.sidebar.selectbox("Race", [
    "White", "Black", "Asian-Pac-Islander", "Amer-Indian-Eskimo", "Other"
])
gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
hours_per_week = st.sidebar.slider("Hours per Week", 1, 99, 40)
fnlwgt = st.sidebar.number_input("Final Weight", min_value=50000, max_value=1000000, value=200000)
capital_gain = st.sidebar.number_input("Capital Gain", min_value=0, max_value=100000, value=0)
capital_loss = st.sidebar.number_input("Capital Loss", min_value=0, max_value=5000, value=0)
native_country = st.sidebar.selectbox("Native Country", [
    "United-States", "India", "Mexico", "Philippines", "Germany", "Canada", "Iran",
    "Vietnam", "China", "Columbia", "Cuba", "England", "France", "Italy", "Japan"
])

# Create dataframe
input_df = pd.DataFrame([{
    'age': age,
    'workclass': workclass,
    'education': education,
    'educational-num': educational_num,
    'occupation': occupation,
    'hours-per-week': hours_per_week,
    'fnlwgt': fnlwgt,
    'capital-gain': capital_gain,
    'capital-loss': capital_loss,
    'relationship': relationship,
    'gender': gender,
    'marital-status': marital_status,
    'native-country': native_country,
    'race': race 
}])

# Show input
st.write("### 🔎 Input Data")
st.write(input_df)

# Predict button
if st.button("Predict Salary Class"):
    prediction = model.predict(input_df)
    st.success(f"✅ Prediction: {'>50K' if prediction[0] == 1 else '≤50K'}")

# Batch prediction section
st.markdown("---")
st.markdown("#### 📂 Batch Prediction")
uploaded_file = st.file_uploader("Upload a CSV file for batch prediction", type="csv")

if uploaded_file is not None:
    batch_data = pd.read_csv(uploaded_file)
    st.write("Uploaded data preview:", batch_data.head())
    batch_preds = model.predict(batch_data)
    batch_data['PredictedClass'] = batch_preds
    st.write("✅ Predictions:")
    st.write(batch_data.head())
    csv = batch_data.to_csv(index=False).encode('utf-8')
    st.download_button("Download Predictions CSV", csv, file_name='predicted_classes.csv', mime='text/csv')

import streamlit as st
import pandas as pd
import torch
import torch.nn as nn
import pickle

# Load model components
with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

with open("imputers.pkl", "rb") as f:
    cat_imputer, num_imputer = pickle.load(f)

with open("reference_columns.pkl", "rb") as f:
    reference_columns = pickle.load(f)

# Define model architecture
class DNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(DNN, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, hidden_size[0]),
            nn.LeakyReLU(),
            nn.Linear(hidden_size[0], hidden_size[1]),
            nn.LeakyReLU(),
            nn.Linear(hidden_size[1], hidden_size[2]),
            nn.LeakyReLU(),
            nn.Linear(hidden_size[2], hidden_size[3]),
            nn.LeakyReLU(),
            nn.Linear(hidden_size[3], hidden_size[4]),
            nn.LeakyReLU(),
            nn.Linear(hidden_size[4], hidden_size[5]),
            nn.LeakyReLU(),
            nn.Linear(hidden_size[5], hidden_size[6]),
            nn.LeakyReLU(),
            nn.Linear(hidden_size[6], output_size)
        )

    def forward(self, x):
        return self.layers(x)

# Instantiate and load model
input_size = len(reference_columns)
hidden_size = [128, 64, 32, 16, 8, 4, 2]
output_size = 2
model = DNN(input_size, hidden_size, output_size)
model.load_state_dict(torch.load("dnn_depression_model.pth", map_location=torch.device("cpu")))
model.eval()

# UI inputs
st.title("Depression Prediction App")

gender = st.selectbox("Gender", ["Male", "Female"])
age = st.number_input("Age", min_value=10, max_value=100, step=1)
role = st.selectbox("Working Professional or Student", ["Working Professional", "Student"])
academic_pressure = st.number_input("Academic Pressure", min_value=0.0, max_value=10.0, step=0.1)
work_pressure = st.number_input("Work Pressure", min_value=0.0, max_value=10.0, step=0.1)
cgpa = st.number_input("CGPA", min_value=0.0, max_value=10.0, step=0.01)
study_satisfaction = st.number_input("Study Satisfaction", min_value=0.0, max_value=10.0, step=0.1)
job_satisfaction = st.number_input("Job Satisfaction", min_value=0.0, max_value=10.0, step=0.1)
sleep_duration = st.selectbox("Sleep Duration", ["Less than 5 hours", "7-8 hours", "More than 8 hours"])
diet = st.selectbox("Dietary Habits", ["Moderate", "Healthy", "Unhealthy"])
suicidal_thoughts = st.selectbox("Have you ever had suicidal thoughts?", ["Yes", "No"])
work_study_hours = st.number_input("Work/Study Hours", min_value=0.0, max_value=24.0, step=0.5)
financial_stress = st.number_input("Financial Stress", min_value=0.0, max_value=10.0, step=0.1)
family_history = st.selectbox("Family History of Mental Illness", ["Yes", "No"])

# On Predict
if st.button("Predict"):
    input_dict = {
        'Gender': gender,
        'Age': age,
        'Working Professional or Student': role,
        'Academic Pressure': academic_pressure,
        'Work Pressure': work_pressure,
        'CGPA': cgpa,
        'Study Satisfaction': study_satisfaction,
        'Job Satisfaction': job_satisfaction,
        'Sleep Duration': sleep_duration,
        'Dietary Habits': diet,
        'Have you ever had suicidal thoughts ?': suicidal_thoughts,
        'Work/Study Hours': work_study_hours,
        'Financial Stress': financial_stress,
        'Family History of Mental Illness': family_history
    }

    input_df = pd.DataFrame([input_dict])

    # Identify column types from training
    categorical_cols = [
        'Gender', 'Working Professional or Student', 'Sleep Duration',
        'Dietary Habits', 'Have you ever had suicidal thoughts ?', 'Family History of Mental Illness'
    ]
    numerical_cols = [col for col in input_df.columns if col not in categorical_cols]

    # Apply imputers
    input_df[categorical_cols] = cat_imputer.transform(input_df[categorical_cols])
    input_df[numerical_cols] = num_imputer.transform(input_df[numerical_cols])

    # One-hot encoding
    input_df = pd.get_dummies(input_df, columns=categorical_cols)

    # Align columns
    for col in reference_columns:
        if col not in input_df.columns:
            input_df[col] = 0
    input_df = input_df[reference_columns]

    # Normalize numerical columns
    input_df[numerical_cols] = scaler.transform(input_df[numerical_cols])
    
     # Ensure all data is numeric
    input_df = input_df.apply(pd.to_numeric, errors='coerce').fillna(0).astype(float)

    # Predict
    input_tensor = torch.tensor(input_df.values, dtype=torch.float32)
    with torch.no_grad():
        output = model(input_tensor)
        pred = torch.argmax(output, dim=1).item()

    st.markdown(f"### Prediction: {'Depressed' if pred == 1 else 'Not Depressed'}")


import streamlit as st
import torch
import pandas as pd
import numpy as np
import joblib

# Load preprocessing tools and model
categorical_cols = joblib.load("categorical_cols.pkl")
numerical_cols = joblib.load("numerical_cols.pkl")
label_encoders = joblib.load("label_encoders.pkl")
scaler = joblib.load("scaler.pkl")
medians = joblib.load("medians.pkl")

class MLPModel(torch.nn.Module):
    def __init__(self, input_dim):
        super(MLPModel, self).__init__()
        self.model = torch.nn.Sequential(
            torch.nn.Linear(input_dim, 64),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.3),
            torch.nn.Linear(64, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, 1),
            torch.nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)

# Load trained model
input_dim = len(categorical_cols) + len(numerical_cols)
model = MLPModel(input_dim)
model.load_state_dict(torch.load("model.pth", map_location=torch.device("cpu")))
model.eval()

st.title(" Depression Prediction Tool")

# Collect user inputs
user_data = {}
for col in categorical_cols:
    user_data[col] = st.selectbox(f"{col}", options=label_encoders[col].classes_)

for col in numerical_cols:
    user_data[col] = st.number_input(f"{col}", value=float(medians[col]))

# Predict on button click
if st.button("Predict Depression"):
    # Prepare input
    input_df = pd.DataFrame([user_data])

    for col in categorical_cols:
        le = label_encoders[col]
        if input_df[col][0] not in le.classes_:
            input_df[col] = 'Unknown'
        if 'Unknown' not in le.classes_:
            le.classes_ = np.append(le.classes_, 'Unknown')
        input_df[col] = le.transform(input_df[col])

    input_df[numerical_cols] = scaler.transform(input_df[numerical_cols])
    input_tensor = torch.tensor(input_df.values, dtype=torch.float32)

    # Predict
    with torch.no_grad():
        output = model(input_tensor)
        prediction = (output >= 0.5).int().item()

    st.success("Prediction: Depressed" if prediction == 1 else "Prediction: Not Depressed")














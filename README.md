Depression Prediction Using Mental Health Survey Data
Overview:
This project is a Depression Prediction Web Application built using Python, Streamlit, and PyTorch. It analyzes mental health survey data to predict whether an individual is likely to experience depression based on various lifestyle, psychological, and demographic factors. The application enables real-time prediction by collecting inputs from users through an intuitive interface, making it suitable for both students and working professionals.

Features:
Real-time Predictions:
Depression Classification: Classifies whether a user is likely to be depressed or not using a pre-trained neural network.

Data Input:
Users can enter personal, lifestyle, and emotional attributes directly into the web interface.

Preprocessing Pipeline:

Missing Value Handling:
Categorical values filled with 'Unknown'
Numerical values filled with median values

Label Encoding:
Categorical variables are encoded using LabelEncoder

Feature Scaling:
Standardization of numerical features using StandardScaler

Consistent Feature Alignment:
Ensures that training and inference features match exactly

Model:
Architecture:
Model: Multi-Layer Perceptron (MLP)

Input Layer: Matches number of selected categorical + numerical features

Hidden Layers:

First Layer: 64 neurons + ReLU + Dropout (0.3)

Second Layer: 32 neurons + ReLU

Output Layer: 1 neuron + Sigmoid (for binary classification)

Model File:
Trained model weights are saved as model.pth
Automatically loaded in the Streamlit application

Technologies Used:
Python
Streamlit
PyTorch
pandas
scikit-learn
joblib (for serialization of encoders, scalers, etc.)

Prerequisites:
Before running the app, ensure the following are installed:
Python >= 3.10

Required libraries: pip install -r requirements.txt

Example requirements.txt:
nginx
Copy
Edit
streamlit
torch
pandas
numpy
scikit-learn
joblib

Usage:
Run the application locally:
bash:"streamlit run streamlit_app.py"

How to Use:
Fill in the appropriate fields shown based on your selection
Click on "Predict Depression"
View the result: "Depressed" or "Not Depressed"

🧠 Model Details:
Training Data: Cleaned and preprocessed using LabelEncoder, StandardScaler

Trained on: Mental Health Survey dataset

Target Variable: Depression (Binary: 1 = Depressed, 0 = Not Depressed)

Project Structure:

├── streamlit_app.py                # Main Streamlit app
├── model.pth                       # Trained PyTorch model
├── categorical_cols.pkl            # Categorical columns used during training
├── numerical_cols.pkl              # Numerical columns used during training
├── label_encoders.pkl              # Encoders for categorical values
├── scaler.pkl                      # Scaler object for numerical normalization
├── medians.pkl                     # Median values used for imputing missing data
├── requirements.txt                # List of dependencies
└── README.md                       # This file

License:
This project is licensed under the MIT License.

Author:
Ramadevi N









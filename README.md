# Depression Prediction from Survey Data

## Overview

This project is a **Depression Prediction System** built using **Python, PyTorch, Scikit-learn, and Streamlit**. It enables users to input personal, demographic, and lifestyle information through a web interface and predicts whether they are likely experiencing depression using a trained Deep Neural Network (DNN). The goal is to provide accessible mental health insights powered by machine learning.

## Features

* **Streamlit-Based Input Interface**: Collects user data via an interactive form (age, education, lifestyle, etc.).
* **Real-Time Depression Prediction**: Predicts the likelihood of depression using a trained PyTorch DNN model.
* **Preprocessing Pipeline**: Includes data imputation, scaling, and encoding based on training-time configuration.
* **Model Evaluation**: Evaluates model using accuracy, precision, recall, and F1-score.

## Technologies Used

* Python
* PyTorch
* Scikit-learn
* Pandas / NumPy
* Streamlit
* Joblib / Pickle

## Dataset

* **Mental Health Survey Data**: Includes demographic, behavioral, and medical history attributes.
* Binary Classification Target: Depression (Yes/No)

## Prerequisites

Ensure you have the following installed:

* Python (>=3.9)
* pip
* Required Python libraries:

```bash
pip install torch scikit-learn pandas streamlit joblib
```

## Usage

**Train the model:**

Run `training.ipynb` to preprocess data, train the DNN, and save the model with the required preprocessing objects.

**Run the Streamlit app:**

```bash
streamlit run app.py
```

Enter user information in the form.

View the depression prediction result and optionally download it.

## DNN Model Architecture

* Input: Numerical and one-hot encoded categorical features
* Hidden Layers: 7 fully connected layers with the following sizes:
  - 128, 64, 32, 16, 8, 4, 2
* Activation: LeakyReLU after each layer
* Output Layer: 2 units (for binary classification: 0 or 1)
* Loss Function: CrossEntropyLoss
* Optimizer: Adam with learning rate = 0.001

## Directory Structure

```
├── train.csv / test.csv             # Input data
├── training.ipynb                   # Script for training and saving the model
├── app.py                           # Streamlit app for real-time prediction
├── dnn_depression_model.pth         # Trained PyTorch model
├── scaler.pkl                       # StandardScaler used in preprocessing
├── imputers.pkl                     # Imputers for missing value handling
├── reference_columns.pkl            # Feature column reference from training
├── test_predictions.csv             # Output predictions from test set
├── README.md                        # Project documentation
```

## Contribution

Feel free to contribute by forking the repository and submitting pull requests.

## License

This project is licensed under the MIT License.

## Author

Ramadevi N
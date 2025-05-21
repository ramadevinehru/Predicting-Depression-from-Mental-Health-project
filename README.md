# Depression Prediction from Survey Data

## Overview

This project is a **Depression Prediction System** built using **Python, PyTorch, Scikit-learn, and Streamlit**. It enables users to input personal, demographic, and lifestyle information through a web interface and predicts whether they are likely experiencing depression using a trained Deep Neural Network (DNN). The goal is to provide accessible mental health insights powered by machine learning.

## Features

* **Streamlit-Based Input Interface**: Collects user data via an interactive form (age, education, lifestyle, etc.).
* **Real-Time Depression Prediction**: Predicts the likelihood of depression using a trained PyTorch DNN model.
* **Preprocessing Pipeline**: Includes data imputation, scaling, and encoding based on training-time configuration.
* **Model Evaluation**: Evaluates model using accuracy, precision, recall, and F1-score.
* **Exportable Results**: Allows users to save prediction outputs as a downloadable CSV.

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

* Python (>=3.1)
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
* 3 Fully Connected Layers (Dense) with LeakyReLU activations
* Dropout layers for regularization
* Output Layer: 1 unit with Sigmoid activation for binary classification

## Directory Structure

```
├── train.csv / test.csv             # Input data
├── train_model.py                   # Script for training and saving the model
├── streamlit_app_3.py               # Streamlit app for real-time prediction
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

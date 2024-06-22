## Customer Churn Prediction

This project aims to predict customer churn using machine learning models. Customer churn prediction helps businesses identify customers who are likely to stop using their services, enabling them to take proactive measures to retain these customers and reduce losses.

## Table of Contents

- [Overview](#overview)
- [Business Understanding](#business-understanding)
- [Data Understanding](#data-understanding)
- [Data Preprocessing](#data-preprocessing)
- [Model Training and Evaluation](#model-training-and-evaluation)
- [API Deployment](#api-deployment)
- [Usage Example](#usage-example)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## Overview

The goal of this project is to develop a model that predicts whether a customer will churn (leave the service) based on historical data. We used the "Customer Churn Dataset" from Kaggle for this purpose.

## Business Understanding

**Objective:** Reduce customer churn to increase revenue and improve customer retention.

**Business Need:** The retail business needs a model to predict which customers are likely to churn so that targeted marketing strategies can be implemented to retain them.

## Data Understanding

### Dataset

We used a publicly available dataset: **"Customer Churn Dataset" from Kaggle**.

### Data Exploration

- Load the dataset and inspect the columns and data types.
- Identify the target variable (`Churn`) and features (e.g., customer demographics, purchase history).
- Check for missing values and handle them.
- Remove duplicates if any.

## Data Preprocessing

- **Handling Missing Values:** Missing values were handled by filling them with the mean for numerical features and the mode for categorical features.
- **Encoding Categorical Variables:** Categorical variables were encoded using OneHotEncoder.
- **Feature Scaling:** Numerical features were scaled using StandardScaler to ensure they are on the same scale.
- **Balancing the Dataset:** Given the imbalance in the dataset, SMOTE (Synthetic Minority Over-sampling Technique) was applied to generate synthetic samples for the minority class.

## Model Training and Evaluation

### Algorithms Used

Three machine learning algorithms were used to train the models:
1. Logistic Regression
2. Decision Tree
3. Random Forest

### Training Process

The models were trained on a balanced dataset to ensure fair evaluation. The dataset was split into training and testing sets using stratified sampling to maintain class distribution.

### Evaluation Metrics

The models were evaluated using the following metrics:
- **Accuracy**
- **Precision**
- **Recall**
- **F1 Score**

### Results

The Random Forest model showed the best performance with the following metrics:
- **Accuracy:** 98.89%
- **Precision:** 99.25%
- **Recall:** 98.52%
- **F1 Score:** 98.88%

## API Deployment

### Saving the Model

The trained Random Forest model was saved using `joblib`:

```python
import joblib

# Save the trained Random Forest model
joblib.dump(rf, 'random_forest_model.pkl')
```

### Streamlit Deployment

A Streamlit application was created to serve the Random Forest model, allowing for real-time predictions.

#### Streamlit App Code

```python
import streamlit as st
import pandas as pd
import joblib

# Load the trained model
model = joblib.load('random_forest_model.pkl')

# Define the Streamlit app
st.title('Customer Churn Prediction')

# Input fields for customer features
gender = st.selectbox('Gender', ['Male', 'Female'])
SeniorCitizen = st.selectbox('Senior Citizen', [0, 1])
Partner = st.selectbox('Partner', ['Yes', 'No'])
Dependents = st.selectbox('Dependents', ['Yes', 'No'])
tenure = st.number_input('Tenure (months)', min_value=0)
PhoneService = st.selectbox('Phone Service', ['Yes', 'No'])
MultipleLines = st.selectbox('Multiple Lines', ['Yes', 'No', 'No phone service'])
InternetService = st.selectbox('Internet Service', ['DSL', 'Fiber optic', 'No'])
OnlineSecurity = st.selectbox('Online Security', ['Yes', 'No', 'No internet service'])
OnlineBackup = st.selectbox('Online Backup', ['Yes', 'No', 'No internet service'])
DeviceProtection = st.selectbox('Device Protection', ['Yes', 'No', 'No internet service'])
TechSupport = st.selectbox('Tech Support', ['Yes', 'No', 'No internet service'])
StreamingTV = st.selectbox('Streaming TV', ['Yes', 'No', 'No internet service'])
StreamingMovies = st.selectbox('Streaming Movies', ['Yes', 'No', 'No internet service'])
Contract = st.selectbox('Contract', ['Month-to-month', 'One year', 'Two year'])
PaperlessBilling = st.selectbox('Paperless Billing', ['Yes', 'No'])
PaymentMethod = st.selectbox('Payment Method', ['Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)'])
MonthlyCharges = st.number_input('Monthly Charges ($)', min_value=0.0)
TotalCharges = st.number_input('Total Charges ($)', min_value=0.0)

# Function to create tenure group based on tenure
def tenure_group(tenure):
    if tenure <= 12:
        return '0-12 months'
    elif tenure <= 24:
        return '12-24 months'
    elif tenure <= 36:
        return '24-36 months'
    elif tenure <= 48:
        return '36-48 months'
    elif tenure <= 60:
        return '48-60 months'
    else:
        return '60+ months'

tenure_group_value = tenure_group(tenure)

# Create a DataFrame for the input features
input_data = pd.DataFrame({
    'gender': [gender],
    'SeniorCitizen': [SeniorCitizen],
    'Partner': [Partner],
    'Dependents': [Dependents],
    'tenure': [tenure],
    'PhoneService': [PhoneService],
    'MultipleLines': [MultipleLines],
    'InternetService': [InternetService],
    'OnlineSecurity': [OnlineSecurity],
    'OnlineBackup': [OnlineBackup],
    'DeviceProtection': [DeviceProtection],
    'TechSupport': [TechSupport],
    'StreamingTV': [StreamingTV],
    'StreamingMovies': [StreamingMovies],
    'Contract': [Contract],
    'PaperlessBilling': [PaperlessBilling],
    'PaymentMethod': [PaymentMethod],
    'MonthlyCharges': [MonthlyCharges],
    'TotalCharges': [TotalCharges],
    'tenure_group': [tenure_group_value]
})

# Convert categorical variables to numeric
input_data['gender'] = input_data['gender'].apply(lambda x: 1 if x == 'Male' else 0)
input_data['Partner'] = input_data['Partner'].apply(lambda x: 1 if x == 'Yes' else 0)
input_data['Dependents'] = input_data['Dependents'].apply(lambda x: 1 if x == 'Yes' else 0)
input_data['PhoneService'] = input_data['PhoneService'].apply(lambda x: 1 if x == 'Yes' else 0)
input_data['MultipleLines'] = input_data['MultipleLines'].apply(lambda x: 2 if x == 'Yes' else (1 if x == 'No' else 0))
input_data['InternetService'] = input_data['InternetService'].apply(lambda x: 2 if x == 'Fiber optic' else (1 if x == 'DSL' else 0))
input_data['OnlineSecurity'] = input_data['OnlineSecurity'].apply(lambda x: 2 if x == 'Yes' else (1 if x == 'No' else 0))
input_data['OnlineBackup'] = input_data['OnlineBackup'].apply(lambda x: 2 if x == 'Yes' else (1 if x == 'No' else 0))
input_data['DeviceProtection'] = input_data['DeviceProtection'].apply(lambda x: 2 if x == 'Yes' else (1 if x == 'No' else 0))
input_data['TechSupport'] = input_data['TechSupport'].apply(lambda x: 2 if x == 'Yes' else (1 if x == 'No' else 0))
input_data['StreamingTV'] = input_data['StreamingTV'].apply(lambda x: 2 if x == 'Yes' else (1 if x == 'No' else 0))
input_data['StreamingMovies'] = input_data['StreamingMovies'].apply(lambda x: 2 if x == 'Yes' else (1 if x == 'No' else 0))
input_data['Contract'] = input_data['Contract'].apply(lambda x: 2 if x == 'Two year' else (1 if x == 'One year' else 0))
input_data['PaperlessBilling'] = input_data['PaperlessBilling'].apply(lambda x: 1 if x == 'Yes' else 0)
input_data['PaymentMethod'] = input_data['PaymentMethod'].apply(lambda x: 3 if x == 'Electronic check' else (2 if x == 'Mailed check' else (1 if x == 'Bank transfer (automatic)' else 0)))
input_data['tenure_group'] = input_data['tenure_group'].apply(lambda x: {'0-12 months': 1, '12-24 months': 2, '24-36 months': 3, '36-48 months': 4, '48-60 months': 5, '60+ months': 6}[x])

# Debug line to check the shape and columns of the input data
st.write("Features provided for prediction:", input_data.columns.tolist(), input_data.shape)



# Predict churn
if st.button('Predict Churn'):
    prediction = model.predict(input_data)
    if prediction[0] == 1:
        st.write('The customer is likely to churn.')
    else:
        st.write('The customer is not likely to churn.')
```

#### Running the Streamlit App

To run the Streamlit app locally, use the following command:

```sh
streamlit run app.py
```

#### Deploying with ngrok

To deploy the app using ngrok, use the following setup:

```python
from pyngrok import ngrok

# Set your ngrok authtoken
ngrok.set_auth_token("your_ngrok_authtoken")

# Start ngrok tunnel
public_url = ngrok.connect(8501)
print(f"Public URL: {public_url}")

# Run the Streamlit app
!streamlit run app.py
```

## Usage Example

### Using the Streamlit App

1. Run the Streamlit app using the command `streamlit run app.py`.
2. Access the app through the provided ngrok URL.
3. Input customer features in the provided fields.
4. Click the 'Predict Churn' button to see the prediction result.

## Results

The Random Forest model outperformed Logistic Regression and Decision Tree models, making it the most reliable choice for predicting customer churn. The balanced dataset ensured that the models were trained on an equal number of examples from both classes, resulting in reliable and unbiased performance metrics.

## Contributing

Contributions are welcome! Please follow these steps to contribute:

1. Fork the repository.
2. Create a new branch (`git checkout -b feature-branch`).
3. Commit your changes (`git commit -m 'Add some feature'`).
4. Push to the branch (`git push origin feature-branch`).
5. Create a new Pull Request.

## License

This project is licensed under the MIT License.

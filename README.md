# Customer Churn Prediction

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


Creating Flask API
A Flask API was created to serve the Random Forest model, allowing for real-time predictions.

from flask import Flask, request, jsonify
import joblib
import pandas as pd

# Load the model
model = joblib.load('random_forest_model.pkl')

# Create Flask app
app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    df = pd.DataFrame(data)
    prediction = model.predict(df)
    return jsonify({'prediction': prediction.tolist()})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)

Usage Example
Using cURL
curl -X POST http://127.0.0.1:5000/predict -H "Content-Type: application/json" -d '{"feature_1": [value1], "feature_2": [value2], ...}'

Using Postman
Set the URL to http://127.0.0.1:5000/predict.
Set the method to POST.
Set the Content-Type header to application/json.
Add the JSON body with feature values.
Send the request and view the response.
Results
The Random Forest model outperformed Logistic Regression and Decision Tree models, making it the most reliable choice for predicting customer churn. The balanced dataset ensured that the models were trained on an equal number of examples from both classes, resulting in reliable and unbiased performance metrics.

Contributing
Contributions are welcome! Please follow these steps to contribute:

Fork the repository.
Create a new branch (git checkout -b feature-branch).
Commit your changes (git commit -m 'Add some feature').
Push to the branch (git push origin feature-branch).
Create a new Pull Request.
License
This project is licensed under the MIT License


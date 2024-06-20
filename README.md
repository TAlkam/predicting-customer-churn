# Customer Churn Prediction

This project aims to predict customer churn using a machine learning model. The project includes data preprocessing, model training, evaluation, and deployment using a Flask API.

## Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Installation](#installation)
- [Usage](#usage)
- [Model Training](#model-training)
- [API Deployment](#api-deployment)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## Overview

Customer churn prediction helps businesses identify customers who are likely to stop using their services. This project uses a Random Forest model to predict churn based on customer data.

## Dataset

The dataset used in this project is synthetic and was generated using `make_classification` from `sklearn.datasets`. The dataset is balanced using SMOTE (Synthetic Minority Over-sampling Technique).

## Installation

1. Clone the repository:

    ```bash
    git clone https://github.com/Talkam/customer-churn-prediction.git
    cd customer-churn-prediction
    ```

2. Create and activate a virtual environment (optional but recommended):

    ```bash
    python3 -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3. Install the required dependencies:

    ```bash
    pip install -r requirements.txt
    ```

## Usage

### Model Training

1. Train the machine learning model and save it:

    ```python
    import joblib
    from sklearn.ensemble import RandomForestClassifier

    # Assuming X_train and y_train are your training data and labels
    rf = RandomForestClassifier()
    rf.fit(X_train, y_train)

    # Save the model
    joblib.dump(rf, 'random_forest_model.pkl')
    ```

### API Deployment

1. Create a Flask API to serve the model:

    ```python
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
    ```

2. Run the Flask API:

    ```bash
    python app.py
    ```

### Sending a Prediction Request

Use tools like Postman or cURL to send POST requests to the API.

Example using cURL:

    ```bash
    curl -X POST http://127.0.0.1:5000/predict -H "Content-Type: application/json" -d '{"feature_1": [value1], "feature_2": [value2], ...}'
    ```

## Results

The model was evaluated using accuracy, precision, recall, and F1 score. The Random Forest model showed the best performance with the following metrics:

- **Accuracy**: 98.89%
- **Precision**: 99.25%
- **Recall**: 98.52%
- **F1 Score**: 98.88%

## Contributing

Contributions are welcome! Please create a pull request or open an issue to discuss your ideas.

1. Fork the repository.
2. Create a new branch (`git checkout -b feature-branch`).
3. Commit your changes (`git commit -m 'Add some feature'`).
4. Push to the branch (`git push origin feature-branch`).
5. Create a new Pull Request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

# Diamond Price Predictor

This project provides a web interface and API to predict the price of diamonds based on various features such as carat, cut, color, clarity, depth, and table. The model used is an Multi Layer Perceptron regressor trained on a diamond dataset.

## Features

- Predict diamond prices using a machine learning model.
- Web interface to input diamond features and get the predicted price.
- RESTful API to integrate the prediction model into other applications.

## Requirements

- Python 3.7+
- FastAPI
- Uvicorn
- Joblib
- Scikit-learn
- Pandas
- Numpy
- Jinja2

## Directories

```graphql
.
├── app.py               # Main FastAPI application
├── pipline.py           # Preprocessing data pipline
├── models               # Directory containing the trained model
│   └── mlp
│       └── diamonds_model_mlp.keras
├── templates            # Directory containing HTML templates
│   └── index.html
├── static               # Directory containing static files (e.g., CSS)
│   └── styles.css
├── requirements.txt     # Project dependencies
└── README.md            # This README file

```

## Installation

1. Clone the repository:

```bash
git clone https://github.com/VicFonch/lapidarist-problem-EDT
cd diamond-price-predictor
```

2. Create a virtual environment and activate it:

```bash
python -m venv venv
source venv/bin/activate # On Windows use `venv\Scripts\activate`
```

3. Install the required dependencies:

```bash
pip install -r requirements.txt
```

Ensure that the trained model is placed in the `models/mlp/` directory.

## Usage

### Running the Web App

```bash
uvicorn app:app --reload
```

Example Request:

```bash
curl -X POST "http://127.0.0.1:8000/predict" \
-H "Content-Type: application/x-www-form-urlencoded" \
-d "carat=0.3&cut=Ideal&color=E&clarity=SI1&depth=61.5&table=55"
```

```json
{
  "prediction": 1125.45
}
```

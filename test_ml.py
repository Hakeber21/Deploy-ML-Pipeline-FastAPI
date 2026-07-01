import pytest
# TODO: add necessary import
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from ml.data import process_data
from ml.model import train_model, compute_model_metrics, inference

cat_features = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]

@pytest.fixture
def data():
    """
    Fixture to load and preprocess the data for testing.
    """
    # Load the data
    df = pd.read_csv("data/census.csv")
    
    # Split the data into train and test sets
    train, _ = train_test_split(df, test_size=0.20, random_state=42)
    
    # Process the training data
    X, y, _, _ = process_data(
        train, categorical_features=cat_features, label="salary", training=True
    )
    
    return X, y

# TODO: implement the first test. Change the function name and input as needed
def test_train_model_returns_random_forest(data):
    """
    Test that the train_model function returns a fitted RandomForestClassifier.
    """
    X, y = data
    model = train_model(X, y)
    assert isinstance(model, RandomForestClassifier), "The model should be a RandomForestClassifier."


# TODO: implement the second test. Change the function name and input as needed
def test_inference_returns_array(data):
    """
    Test that the inference function returns a numpy array with one prediction per row, all 0 or 1.
    """
    X, y = data
    model = train_model(X, y)
    predictions = inference(model, X)
    assert isinstance(predictions, np.ndarray), "The predictions should be a numpy array."
    assert len(predictions) == len(X), "The number of predictions should match the number of input samples."
    assert set(np.unique(predictions)).issubset({0, 1}), "The predictions should only contain 0 or 1."


# TODO: implement the third test. Change the function name and input as needed
def test_compute_model_metrics_returns_floats():
    """
    # Test that the compute_model_metrics function returns three floats: precision, recall, and fbeta between 0 and 1.
    """
    y = np.array([0, 1, 0, 1, 1])
    preds = np.array([0, 1, 0, 0, 1])
    precision, recall, fbeta = compute_model_metrics(y, preds)
    for metric in [precision, recall, fbeta]:
        assert isinstance(metric, float), "The metric should be a float."
        assert 0.0 <= metric <= 1.0, "The metric should be between 0 and 1."

import pytest
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from ml.model import train_model, compute_model_metrics, inference

def test_train_model_algorithm_type():
    """
    Test that the train_model function successfully builds and returns 
    the expected RandomForestClassifier algorithm.
    """
    # Create simple dummy data
    X_train = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
    y_train = np.array([0, 1, 0, 1])
    
    # Train the model
    model = train_model(X_train, y_train)
    
    # Assert the model is the correct type and has a predict method (is fitted)
    assert isinstance(model, RandomForestClassifier)
    assert hasattr(model, "predict")


def test_compute_model_metrics_accuracy():
    """
    Test that the compute_model_metrics function correctly calculates 
    precision, recall, and the F1 score using known inputs.
    """
    # Define known true labels and predictions
    # True Positives: 1, False Positives: 0, False Negatives: 1, True Negatives: 2
    y_true = np.array([1, 1, 0, 0])
    preds = np.array([1, 0, 0, 0])
    
    # Calculate metrics
    precision, recall, fbeta = compute_model_metrics(y_true, preds)
    
    # Precision = 1 / (1+0) = 1.0
    assert precision == 1.0
    # Recall = 1 / (1+1) = 0.5
    assert recall == 0.5
    # F1 = 2 * (1.0 * 0.5) / (1.0 + 0.5) = 0.666...
    assert round(fbeta, 2) == 0.67


def test_inference_returns_expected_shape():
    """
    Test that the inference function returns predictions in the expected format
    and that the number of predictions matches the number of input rows.
    """
    # Set up a dummy fitted model
    X_train = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
    y_train = np.array([0, 1, 0, 1])
    model = RandomForestClassifier(n_estimators=10, random_state=42)
    model.fit(X_train, y_train)
    
    # Create test data with 2 rows
    X_test = np.array([[2, 3], [6, 7]])
    
    # Run inference
    preds = inference(model, X_test)
    
    # Assert we get exactly 2 predictions back and it is a numpy array
    assert len(preds) == 2
    assert isinstance(preds, np.ndarray)

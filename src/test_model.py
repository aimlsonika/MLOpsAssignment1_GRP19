"""
This script tests a logistic regression model with Optuna on the Diabetes dataset.
"""
import os
import joblib
from src.model import train_model_with_optuna, preprocess_data

def test_train_model_with_optuna():
    """
    Tests the model training function with Optuna hyperparameter tuning.
    Asserts if the accuracy of the best model on the test data is < 70%...
    """
    # Run the training with Optuna
    train_model_with_optuna()

    # Load the saved model and verify performance
    model_path = "modelfinal_best.joblib"
    assert os.path.exists(model_path), "Best model file was not saved!"

    # Load the saved model and verify
    model = joblib.load(model_path)

    # Optionally, load the test data to verify accuracy
    _, x_test, _, y_test = preprocess_data()

    # Evaluate the model
    accuracy = model.score(x_test, y_test)
    assert accuracy > 0.7, f"Model accuracy is too low: {accuracy:.2f}"

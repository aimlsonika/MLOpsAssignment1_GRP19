"""
This script trains a logistic regression model on the Diabetes dataset.
"""
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, mean_squared_error
import mlflow
import mlflow.sklearn
import joblib
import optuna


def preprocess_data():
    """
    Load and preprocess the Diabetes dataset.
    Returns Preprocessed training and test data.
    """
    file_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'diabetes.csv')
    data = pd.read_csv(os.path.abspath(file_path))
    x = data.drop('Outcome', axis=1)
    y = data['Outcome']
    x_train, x_test, y_train, y_test = \
        train_test_split(x, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)
    return x_train, x_test, y_train, y_test


def objective(trial):
    """
    Optuna objective function for hyperparameter optimization.
    """
    # Use preprocessed data
    x_train, x_test, y_train, y_test = preprocess_data()

    # Suggest hyperparameters
    c_param = trial.suggest_float("C", 0.01, 10.0, log=True)
    max_iter = trial.suggest_int("max_iter", 100, 1000, step=100)

    # Train the model
    model = LogisticRegression(C=c_param, max_iter=max_iter)
    model.fit(x_train, y_train)

    # Evaluate the model
    y_pred = model.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)

    # Log to MLflow
    with mlflow.start_run(nested=True):
        mlflow.log_param("C", c_param)
        mlflow.log_param("max_iter", max_iter)
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("mse", mse)
        mlflow.sklearn.log_model(model, artifact_path="model")

    return accuracy


def train_model_with_optuna():
    """
    Trains the model with Optuna hyperparameter tuning.
    Logs metrics and parameters using MLflow.
    """
    # Preprocess the data
    x_train, _, y_train, _ = preprocess_data()

    mlflow.set_tracking_uri("http://127.0.0.1:8000")
    mlflow.set_experiment("Diabetes Prediction with Optuna")
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=3)  # Run 3 trials for demonstration

    print("Best Trial:")
    print(f"  Value: {study.best_trial.value}")
    print(f"  Params: {study.best_trial.params}")

    # Save the best model
    best_model = LogisticRegression(**study.best_trial.params)
    best_model.fit(x_train, y_train)
    joblib.dump(best_model, "best_model.joblib")


if __name__ == "__main__":
    train_model_with_optuna()

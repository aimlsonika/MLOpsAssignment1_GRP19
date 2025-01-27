"""
This is a Flask application for predicting diabetes based on input data.
"""
import logging
import warnings
import pandas as pd
from flask import Flask, request, jsonify
import joblib

# Suppress warnings and configure logging
# Suppress sklearn deprecation warnings
warnings.filterwarnings("ignore", message=".*_IS_DEPRECATED_PICKLE.*")
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Initialize Flask application
app = Flask(__name__)

# Load the trained model
try:
    model = joblib.load("modelfinal_best.joblib")
    logging.info("Model successfully loaded.")
except FileNotFoundError:
    logging.error("Model file not found. Please ensure the model file is in the specified path.")
    raise
except joblib.externals.loky.process_executor.TerminatedWorkerError as e:
    logging.error("Error loading model: %s",e)
    raise

@app.route('/test', methods=['GET'])
def diabetes_test():
    """
    Endpoint to predict diabetes based on user input.
    """
    return "123"

@app.route('/diabetes_prediction', methods=['POST'])
def diabetes_prediction():
    """
    Endpoint to predict diabetes based on user input.
    """
    try:
        # Parse input data
        data = request.json
        if not data or "data" not in data:
            return jsonify({"error": "Invalid input. \
                            Please provide data in JSON format."}), 400
        # Create a DataFrame from the input data
        df = pd.DataFrame(data["data"])

        # Check if required columns are present
        required_columns = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
                            'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
        if not all(column in df.columns for column in required_columns):
            return jsonify({"error": "Missing required columns. \
                            Expected: %s"},required_columns), 400
        # Convert data types to match the model's requirements
        df = df[required_columns]
        df = df.astype({
            'Pregnancies': 'int64',
            'Glucose': 'int64',
            'BloodPressure': 'int64',
            'SkinThickness': 'int64',
            'Insulin': 'int64',
            'BMI': 'float64',
            'DiabetesPedigreeFunction': 'float64',
            'Age': 'int64'
        })

        # Make predictions
        predictions = model.predict(df)
        final_predictions = pd.DataFrame(
            list(predictions),
            columns=["Based on your test report, the diabetes result is:"]
        ).to_dict(orient="records")

        return jsonify(final_predictions), 200

    except pd.errors.EmptyDataError:
        logging.error("Received empty data.")
        raise
        #return jsonify({"error": "Input data is empty."}), 400
    except FileNotFoundError as e:
        logging.error("File not found: %s", e)
        raise
        #return jsonify({"error": "File not found."}), 404
    except ValueError as e:
        logging.error("Value error during prediction: %s",e)
        raise
        #return jsonify({"error": "Invalid data provided: %s"},e), 400
    except Exception as e:  # pylint: disable=W0718
        logging.error("Unexpected error: %s",e)
        raise
        #return jsonify({"error": "An unexpected error occurred."}), 500

if __name__ == "__main__":
    try:
        logging.info("Starting Flask app...")
        app.run(debug=True, host="0.0.0.0", port=5001)
    except Exception as e:
        logging.error("Failed to start Flask app: %s",e)
        raise

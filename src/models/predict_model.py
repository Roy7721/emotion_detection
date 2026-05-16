import pandas as pd
import numpy as np

from sklearn.metrics import (
    precision_score,
    accuracy_score,
    recall_score,
    roc_auc_score
)

import pickle
import json
import os
import logging


# =========================
# Logging Configuration
# =========================

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger("model_evaluation")

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)

file_handler = logging.FileHandler('model_evaluation.log')
file_handler.setLevel(logging.DEBUG)

logger.addHandler(console_handler)
logger.addHandler(file_handler)


# =========================
# Import Model
# =========================

def import_model(model_path: str):

    try:

        logger.info("Loading trained model")

        with open(model_path, 'rb') as f:
            model = pickle.load(f)

        logger.info("Model loaded successfully")

        return model

    except Exception as e:
        logger.error(f"Error loading model: {e}")
        raise


# =========================
# Load Test Data
# =========================

def load_test_data(test_data_path: str):

    try:

        logger.info("Loading test feature data")

        test_df = pd.read_csv(
            os.path.join(test_data_path, 'test_bow(gb).csv')
        )

        x_test = test_df.drop('label', axis=1)

        y_test = test_df['label']

        logger.info("Test data loaded successfully")

        return x_test, y_test

    except Exception as e:
        logger.error(f"Error loading test data: {e}")
        raise


# =========================
# Prediction
# =========================

def predict(model, x_test):

    try:

        logger.info("Generating predictions")

        y_pred = model.predict(x_test)

        logger.info("Prediction completed successfully")

        return y_pred

    except Exception as e:
        logger.error(f"Error during prediction: {e}")
        raise


# =========================
# Model Evaluation
# =========================

def evaluate_model(y_test, y_pred):

    try:

        logger.info("Evaluating model performance")

        accuracy = accuracy_score(y_test, y_pred)

        precision = precision_score(
            y_test,
            y_pred,
            average='weighted'
        )

        recall = recall_score(
            y_test,
            y_pred,
            average='weighted'
        )

        logger.info(
            f"Accuracy: {accuracy}"
        )

        logger.info(
            f"Precision: {precision}"
        )

        logger.info(
            f"Recall: {recall}"
        )

        logger.info("Model evaluation completed")

        return accuracy, precision, recall

    except Exception as e:
        logger.error(f"Error evaluating model: {e}")
        raise


# =========================
# Save Metrics
# =========================

def save_metrics(metrics_dict):

    try:

        logger.info("Saving evaluation metrics")

        with open('metrics.json', 'w') as f:
            json.dump(metrics_dict, f, indent=4)

        logger.info("Metrics saved successfully")

    except Exception as e:
        logger.error(f"Error saving metrics: {e}")
        raise


# =========================
# Main Function
# =========================

def main():

    try:

        logger.info("Model evaluation pipeline started")

        model_path = os.path.join('models', 'model.pkl')

        test_data_path = os.path.join(
            'data',
            'interim'
        )

        model = import_model(model_path)

        x_test, y_test = load_test_data(
            test_data_path
        )

        y_pred = predict(model, x_test)

        accuracy, precision, recall = evaluate_model(
            y_test,
            y_pred
        )

        metrics_dict = {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall
        }

        save_metrics(metrics_dict)

        logger.info("Model evaluation pipeline completed")

    except Exception as e:
        logger.error(f"Pipeline failed: {e}")


if __name__ == "__main__":
    main()
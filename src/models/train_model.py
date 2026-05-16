import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import pickle
import yaml
import os
import logging
from dvclive import Live

live = Live(save_dvc_exp=True)

# =========================
# Logging Configuration
# =========================

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger("model_building")

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)

file_handler = logging.FileHandler('model_building.log')
file_handler.setLevel(logging.DEBUG)

logger.addHandler(console_handler)
logger.addHandler(file_handler)


# =========================
# Load Training Data
# =========================

def load_data(data_path: str):

    try:

        logger.info("Loading training feature data")

        train_df = pd.read_csv(
            os.path.join(data_path, 'train_bow(rf).csv')
        )

        logger.info("Training data loaded successfully")

        return train_df

    except Exception as e:
        logger.error(f"Error loading training data: {e}")
        raise


# =========================
# Train Model
# =========================

def train_model(
    train_df: pd.DataFrame,
    params_path: str
):

    try:

        logger.info("Starting model training")

        logger.info("Separating features and labels")

        x_train = train_df.drop('label', axis=1)

        y_train = train_df['label']

        logger.info("Loading model parameters")

        yaml_content = yaml.safe_load(
            open(params_path, 'r')
        )
        live.log_params({
            "n_estimators": yaml_content['model_building']['n_estimators']
        })
        n_estimators = yaml_content[
            'model_building'
        ]['n_estimators']

        logger.info(
            f"Initializing RandomForestClassifier with n_estimators={n_estimators}"
        )

        model = RandomForestClassifier(
            n_estimators=n_estimators,
            random_state=42
        )

        logger.info("Training Random Forest model")

        model.fit(x_train, y_train)

        logger.info("Model training completed successfully")

        return model

    except Exception as e:
        logger.error(f"Error during model training: {e}")
        raise


# =========================
# Save Model
# =========================

def save_model(model, model_path: str):

    try:

        logger.info("Saving trained model")

        with open(model_path, 'wb') as f:
            pickle.dump(model, f)

        logger.info("Model saved successfully")

    except Exception as e:
        logger.error(f"Error saving model: {e}")
        raise


# =========================
# Main Function
# =========================

def main():

    try:

        logger.info("Model building pipeline started")

        data_path = os.path.join(
            'data',
            'interim'
        )

        params_path = os.path.join(
            'params.yaml'
        )

        model_path = os.path.join(
            'models',
            'model.pkl'
        )

        train_df = load_data(data_path)

        model = train_model(
            train_df,
            params_path
        )

        save_model(
            model,
            model_path
        )

        logger.info("Model building pipeline completed")

    except Exception as e:
        logger.error(f"Pipeline failed: {e}")


if __name__ == "__main__":
    main()
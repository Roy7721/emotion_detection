import numpy as np
import pandas as pd
import yaml
import os
import logging

from sklearn.feature_extraction.text import CountVectorizer


# =========================
# Logging Configuration
# =========================

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger("feature_engineering")

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)

file_handler = logging.FileHandler('feature_engineering.log')
file_handler.setLevel(logging.DEBUG)

logger.addHandler(console_handler)
logger.addHandler(file_handler)


# =========================
# Load Data
# =========================

def load_data(data_path: str):

    try:
        logger.info("Loading processed training and testing data")

        train_df = pd.read_csv(
            os.path.join(data_path, 'train_processed.csv')
        )

        test_df = pd.read_csv(
            os.path.join(data_path, 'test_processed.csv')
        )

        logger.info("Processed data loaded successfully")

        return train_df, test_df

    except Exception as e:
        logger.error(f"Error loading processed data: {e}")
        raise


# =========================
# Feature Engineering
# =========================

def feature_engineering(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    params_path: str
):

    try:

        logger.info("Starting feature engineering")

        logger.info("Dropping null values")
        train_df.dropna(subset=['content'], inplace=True)
        test_df.dropna(subset=['content'], inplace=True)

        logger.info("Converting content column to string")
        train_df['content'] = train_df['content'].astype(str)
        test_df['content'] = test_df['content'].astype(str)

        logger.info("Separating features and labels")

        X_train = train_df['content'].values
        y_train = train_df['sentiment'].values

        X_test = test_df['content'].values
        y_test = test_df['sentiment'].values

        logger.info("Loading parameters from params.yaml")

        yaml_content = yaml.safe_load(
            open(params_path, 'r')
        )

        max_features = yaml_content[
            'feature_engineering'
        ]['max_features']

        logger.info(
            f"Initializing CountVectorizer with max_features={max_features}"
        )

        vectorizer = CountVectorizer(
            max_features=max_features
        )

        logger.info("Applying Bag of Words on training data")

        X_train_bow = vectorizer.fit_transform(X_train)

        logger.info("Transforming testing data")

        X_test_bow = vectorizer.transform(X_test)

        logger.info("Converting sparse matrix to DataFrame")

        train_df = pd.DataFrame(
            X_train_bow.toarray()
        )

        train_df['label'] = y_train

        test_df = pd.DataFrame(
            X_test_bow.toarray()
        )

        test_df['label'] = y_test

        logger.info("Feature engineering completed successfully")

        return train_df, test_df

    except Exception as e:
        logger.error(f"Error during feature engineering: {e}")
        raise


# =========================
# Save Features
# =========================

def save_features(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    data_processed_path: str
):

    try:

        logger.info("Saving engineered features")

        os.makedirs(data_processed_path, exist_ok=True)

        train_df.to_csv(
            os.path.join(data_processed_path, 'train_bow(gb).csv'),
            index=False
        )

        test_df.to_csv(
            os.path.join(data_processed_path, 'test_bow(gb).csv'),
            index=False
        )

        logger.info("Feature files saved successfully")

    except Exception as e:
        logger.error(f"Error saving feature files: {e}")
        raise


# =========================
# Main Function
# =========================

def main():

    try:

        logger.info("Feature engineering pipeline started")

        data_path = os.path.join('data', 'processed')

        train_df, test_df = load_data(data_path)

        params_path = os.path.join('params.yaml')

        train_df, test_df = feature_engineering(
            train_df,
            test_df,
            params_path
        )

        save_features(
            train_df,
            test_df,
            os.path.join('data', 'interim')
        )

        logger.info("Feature engineering pipeline completed")

    except Exception as e:
        logger.error(f"Pipeline failed: {e}")


if __name__ == "__main__":
    main()
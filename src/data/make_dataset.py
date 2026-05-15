from os import path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import yaml
import os
import logging

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

logger = logging.getLogger("make_dataset")

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)
logger.addHandler(console_handler)

file_handler = logging.FileHandler('make_dataset.log')
file_handler.setLevel(logging.DEBUG)
logger.addHandler(file_handler)


def load_param(params_path: str) -> float:
    try:
        with open(params_path) as f:
            params = yaml.safe_load(f)
            logger.info(f"Parameters loaded successfully from {params_path}")
            test_size = params['make_dataset']['test_size']
            logger.debug(f"Test size parameter: {test_size}")
        return test_size

    except FileNotFoundError:
        logger.error(f"Parameter file not found: {params_path}")

    except KeyError:
        logger.error("Parameter 'make_dataset.test_size' missing in params.yaml.")


def load_data(data_path: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(data_path)
        return df

    except Exception as e:
        logger.error(f"Error loading data: {e}")


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    try:
        df.drop(columns=['tweet_id'], inplace=True)

        final_df = df[df['sentiment'].isin(['happiness', 'sadness'])]

        final_df['sentiment'].replace(
            {'happiness': 1, 'sadness': 0},
            inplace=True
        )

        return final_df

    except KeyError as e:
        logger.error(f"Column missing: {e}")


def save_data(
    data_path: str,
    train_data: pd.DataFrame,
    test_data: pd.DataFrame
) -> None:

    try:
        os.makedirs(data_path, exist_ok=True)

        train_data.to_csv(
            os.path.join(data_path, 'train.csv'),
            index=False
        )

        test_data.to_csv(
            os.path.join(data_path, 'test.csv'),
            index=False
        )

    except Exception as e:
        logger.error(f"Error saving data: {e}")


def main():

    test_size = load_param('./params.yaml')

    df = load_data(
        'https://raw.githubusercontent.com/campusx-official/jupyter-masterclass/main/tweet_emotions.csv'
    )

    df = preprocess_data(df)

    train_data, test_data = train_test_split(
        df,
        test_size=test_size,
        random_state=42
    )

    data_path = os.path.join('data', 'raw')

    save_data(data_path, train_data, test_data)


if __name__ == "__main__":
    main()
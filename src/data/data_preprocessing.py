import numpy as np
import re
import pandas as pd
import nltk
import string
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import os
import logging


# =========================
# Logging Configuration
# =========================

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger("data_preprocessing")

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)

file_handler = logging.FileHandler('data_preprocessing.log')
file_handler.setLevel(logging.DEBUG)

logger.addHandler(console_handler)
logger.addHandler(file_handler)


# =========================
# Load Data
# =========================

def load_data(data_path: str):

    try:
        logger.info("Loading training and testing data")

        train_df = pd.read_csv(
            os.path.join(data_path, 'train.csv')
        )

        test_df = pd.read_csv(
            os.path.join(data_path, 'test.csv')
        )

        logger.info("Data loaded successfully")

        return train_df, test_df

    except Exception as e:
        logger.error(f"Error loading data: {e}")
        raise


# =========================
# Download NLTK Resources
# =========================

try:
    logger.info("Downloading NLTK resources")

    nltk.download('wordnet')
    nltk.download('stopwords')

    logger.info("NLTK resources downloaded successfully")

except Exception as e:
    logger.error(f"Error downloading NLTK resources: {e}")


# =========================
# Text Cleaning Functions
# =========================

def lemmatization(text):

    try:
        lemmatizer = WordNetLemmatizer()

        text = text.split()

        text = [
            lemmatizer.lemmatize(word)
            for word in text
        ]

        return " ".join(text)

    except Exception as e:
        logger.error(f"Error in lemmatization: {e}")
        return text


def remove_stop_words(text):

    try:
        stop_words = set(stopwords.words("english"))

        text = [
            word for word in str(text).split()
            if word not in stop_words
        ]

        return " ".join(text)

    except Exception as e:
        logger.error(f"Error removing stopwords: {e}")
        return text


def removing_numbers(text):

    try:
        text = ''.join(
            [char for char in text if not char.isdigit()]
        )

        return text

    except Exception as e:
        logger.error(f"Error removing numbers: {e}")
        return text


def lower_case(text):

    try:
        text = text.split()

        text = [word.lower() for word in text]

        return " ".join(text)

    except Exception as e:
        logger.error(f"Error converting to lowercase: {e}")
        return text


def removing_punctuations(text):

    try:
        text = re.sub(
            '[%s]' % re.escape(string.punctuation),
            ' ',
            text
        )

        text = text.replace('؛', "")

        text = re.sub('\s+', ' ', text).strip()

        return text

    except Exception as e:
        logger.error(f"Error removing punctuations: {e}")
        return text


def removing_urls(text):

    try:
        url_pattern = re.compile(
            r'https?://\S+|www\.\S+'
        )

        return url_pattern.sub(r'', text)

    except Exception as e:
        logger.error(f"Error removing URLs: {e}")
        return text


# =========================
# Normalize Text
# =========================

def normalize_text(df):

    try:
        logger.info("Starting text normalization")

        logger.info("Converting text to lowercase")
        df.content = df.content.apply(
            lambda content: lower_case(content)
        )

        logger.info("Removing stopwords")
        df.content = df.content.apply(
            lambda content: remove_stop_words(content)
        )

        logger.info("Removing numbers")
        df.content = df.content.apply(
            lambda content: removing_numbers(content)
        )

        logger.info("Removing punctuations")
        df.content = df.content.apply(
            lambda content: removing_punctuations(content)
        )

        logger.info("Removing URLs")
        df.content = df.content.apply(
            lambda content: removing_urls(content)
        )

        logger.info("Applying lemmatization")
        df.content = df.content.apply(
            lambda content: lemmatization(content)
        )

        logger.info("Text normalization completed")

        return df

    except Exception as e:
        logger.error(f"Error during text normalization: {e}")
        raise


# =========================
# Save Processed Data
# =========================

def save_processed_data(
    data_path: str,
    train_data: pd.DataFrame,
    test_data: pd.DataFrame
):

    try:
        logger.info("Saving processed data")

        os.makedirs(data_path, exist_ok=True)

        train_data.to_csv(
            os.path.join(data_path, 'train_processed.csv'),
            index=False
        )

        test_data.to_csv(
            os.path.join(data_path, 'test_processed.csv'),
            index=False
        )

        logger.info("Processed data saved successfully")

    except Exception as e:
        logger.error(f"Error saving processed data: {e}")
        raise


# =========================
# Main Function
# =========================

def main():

    try:
        logger.info("Data preprocessing pipeline started")

        data_path = os.path.join('data', 'raw')

        train_df, test_df = load_data(data_path)

        logger.info("Processing training data")
        train_processed_data = normalize_text(train_df)

        logger.info("Processing testing data")
        test_processed_data = normalize_text(test_df)

        save_processed_data(
            os.path.join('data', 'processed'),
            train_processed_data,
            test_processed_data
        )

        logger.info("Data preprocessing pipeline completed")

    except Exception as e:
        logger.error(f"Pipeline failed: {e}")


if __name__ == "__main__":
    main()
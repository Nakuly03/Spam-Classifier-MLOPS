import pandas as pd
import os
from sklearn.model_selection import train_test_split
import logging

log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)

logger = logging.getLogger("data_ingestion")
logger.setLevel(logging.DEBUG)

console_handler = logging.StreamHandler()
file_handler = logging.FileHandler(os.path.join(log_dir, "data_ingestion.log"))

formatter = logging.Formatter(
    "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)


class DataIngestion:

    def __init__(self):
        self.data_path = "experiments/spam.csv"
        self.train_path = "data/interim/train.csv"
        self.test_path = "data/interim/test.csv"

    def load_data(self):
        logger.info("Loading dataset...")
        df = pd.read_csv(self.data_path, encoding="latin1")
        return df

    def preprocess_data(self, df):

        df.drop(columns=["Unnamed: 2", "Unnamed: 3", "Unnamed: 4"], inplace=True)
        df.rename(columns={"v2": "messages", "v1": "label"}, inplace=True)

        return df

    def save_data(self, train_df, test_df):

        os.makedirs("data/interim", exist_ok=True)

        train_df.to_csv(self.train_path, index=False)
        test_df.to_csv(self.test_path, index=False)

        logger.info("Train and test data saved.")

    def run(self):

        df = self.load_data()

        df = self.preprocess_data(df)

        train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

        self.save_data(train_df, test_df)

        return self.train_path, self.test_path
import pandas as pd
import os
import logging
from sklearn.preprocessing import LabelEncoder
import nltk
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
import string
import re

log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)

logger = logging.getLogger("data_preprocessing")
logger.setLevel(logging.DEBUG)

console_handler = logging.StreamHandler()
file_handler = logging.FileHandler(os.path.join(log_dir, "data_preprocessing.log"))

formatter = logging.Formatter(
    "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)


nltk.download("stopwords")

stop_words = set(stopwords.words("english"))
ps = PorterStemmer()

chat_words = {
    "afk": "away from keyboard",
    "asap": "as soon as possible",
    "brb": "be right back",
    "btw": "by the way",
    "fyi": "for your information",
    "gr8": "great",
    "idk": "i do not know",
    "lol": "laughing out loud",
    "lmao": "laughing my ass off",
    "nvm": "never mind",
    "rofl": "rolling on the floor laughing",
    "tbh": "to be honest",
    "thx": "thank you",
    "ttyl": "talk to you later",
    "u": "you",
    "u2": "you too",
    "wtf": "what the fuck",
}


class DataPreprocessing:

    def clean_text(self, message):

        message = message.lower()

        message = re.sub(r"<.*?>", " ", message)

        message = re.sub(r"https?://\S+|www\.\S+", " ", message)

        message = message.translate(str.maketrans("", "", string.punctuation))

        words = []
        for w in message.split():
            if w in chat_words:
                words.append(chat_words[w])
            else:
                words.append(w)

        message = " ".join(words)

        words = []
        for w in message.split():
            if w not in stop_words:
                words.append(w)

        message = " ".join(words)

        words = []
        for w in message.split():
            words.append(ps.stem(w))

        message = " ".join(words)

        message = re.sub(r"\s+", " ", message).strip()

        return message


    def preprocess_df(self, df, text_column="messages", target_column="label"):

        try:

            logger.debug("Encoding label column")

            encoder = LabelEncoder()

            df[target_column] = encoder.fit_transform(df[target_column])

            df = df.drop_duplicates()

            df.loc[:, text_column] = df[text_column].apply(self.clean_text)

            logger.debug("Text preprocessing complete")

            return df

        except Exception as e:

            logger.error("Error during preprocessing: %s", e)

            raise


    def run(self, train_path, test_path):

        train_df = pd.read_csv(train_path)

        test_df = pd.read_csv(test_path)

        train_processed = self.preprocess_df(train_df)

        test_processed = self.preprocess_df(test_df)

        os.makedirs("data/interim", exist_ok=True)

        train_processed_path = "data/interim/train_processed.csv"

        test_processed_path = "data/interim/test_processed.csv"

        train_processed.to_csv(train_processed_path, index=False)

        test_processed.to_csv(test_processed_path, index=False)

        logger.debug("Processed data saved")

        return train_processed_path, test_processed_path
import pandas as pd
import os
import logging
import pickle
from sklearn.feature_extraction.text import CountVectorizer

log_dir = 'logs'
os.makedirs(log_dir, exist_ok=True)

logger = logging.getLogger('feature_engineering')
logger.setLevel(logging.DEBUG)

console_handler = logging.StreamHandler()
file_handler = logging.FileHandler(os.path.join(log_dir, 'feature_engineering.log'))

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)


class FeatureEngineering:

    def __init__(self):
        self.max_features = 50
        self.train_path = './data/interim/train_processed.csv'
        self.test_path = './data/interim/test_processed.csv'

    def load_data(self, file_path):

        try:
            df = pd.read_csv(file_path)
            df.fillna('', inplace=True)
            logger.debug('Data loaded and empty values filled %s', file_path)
            return df

        except Exception as e:
            logger.error('Error loading data: %s', e)
            raise


    def apply_bow(self, train_data, test_data):

        try:

            vectorizer = CountVectorizer(
                max_features=self.max_features,
                ngram_range=(1,2)
            )

            X_train = train_data['messages'].values
            y_train = train_data['label'].values

            X_test = test_data['messages'].values
            y_test = test_data['label'].values

            X_train_bow = vectorizer.fit_transform(X_train)
            X_test_bow = vectorizer.transform(X_test)

            logger.debug('Bag of words transformation complete')

            return X_train_bow, X_test_bow, y_train, y_test, vectorizer

        except Exception as e:
            logger.error('Error during BOW transformation: %s', e)
            raise


    def save_vectorizer(self, vectorizer):

        try:
            os.makedirs('models', exist_ok=True)
            pickle.dump(vectorizer, open('models/vectorizer.pkl', 'wb'))
            logger.debug("Vectorizer saved")

        except Exception as e:
            logger.error("Error saving vectorizer: %s", e)
            raise


    def run(self):

        train_data = self.load_data(self.train_path)
        test_data = self.load_data(self.test_path)

        X_train, X_test, y_train, y_test, vectorizer = self.apply_bow(train_data, test_data)

        self.save_vectorizer(vectorizer)

        return X_train, X_test, y_train, y_test
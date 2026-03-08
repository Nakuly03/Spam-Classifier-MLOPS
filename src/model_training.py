import numpy as np
import os
import logging
import pickle
import mlflow
import mlflow.sklearn

mlflow.set_tracking_uri("file:./mlruns")
mlflow.set_experiment("spam_classifier")

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)

logger = logging.getLogger("model_training")
logger.setLevel(logging.DEBUG)

console_handler = logging.StreamHandler()
file_handler = logging.FileHandler(os.path.join(log_dir, "model_training.log"))

formatter = logging.Formatter(
    "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)


class ModelTraining:

    def __init__(self):

        self.params = {
            "n_estimators": 50,
            "random_state": 42
        }

    def train_model(self, X_train: np.ndarray, y_train: np.ndarray):

        try:

            logger.info("Initializing RandomForest model")

            model = RandomForestClassifier(
                n_estimators=self.params["n_estimators"],
                random_state=self.params["random_state"]
            )

            logger.info("Starting model training")

            model.fit(X_train, y_train)

            logger.info("Model training completed")

            return model

        except Exception as e:

            logger.error("Error during model training: %s", e)

            raise


    def save_model(self, model):

        try:

            os.makedirs("models", exist_ok=True)

            model_path = "models/model.pkl"

            with open(model_path, "wb") as file:
                pickle.dump(model, file)

            logger.info("Model saved at %s", model_path)

        except Exception as e:

            logger.error("Error saving model: %s", e)

            raise


    def run(self, X_train, y_train):

        try:

            logger.info("Starting MLflow experiment")

            mlflow.set_tracking_uri("file:./mlruns")
            mlflow.set_experiment("spam_classifier_experiment")
            
            with mlflow.start_run():

                model = self.train_model(X_train, y_train)

                predictions = model.predict(X_train)

                accuracy = accuracy_score(y_train, predictions)

                mlflow.log_param("n_estimators", self.params["n_estimators"])
                mlflow.log_param("random_state", self.params["random_state"])

                mlflow.log_metric("train_accuracy", accuracy)

                mlflow.sklearn.log_model(model, "spam_classifier_model")

                logger.info("MLflow logging completed")

                self.save_model(model)

            return model

        except Exception as e:

            logger.error("Training pipeline failed: %s", e)

            raise
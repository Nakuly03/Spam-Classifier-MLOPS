import numpy as np
import os
import logging
import json
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score

log_dir = 'logs'
os.makedirs(log_dir, exist_ok=True)

logger = logging.getLogger('model_evaluation')
logger.setLevel(logging.DEBUG)

console_handler = logging.StreamHandler()
file_handler = logging.FileHandler(os.path.join(log_dir, 'model_evaluation.log'))

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)


class ModelEvaluation:

    def evaluate_model(self, model, X_test: np.ndarray, y_test: np.ndarray):

        try:
            y_pred = model.predict(X_test)

            if hasattr(model, "predict_proba"):
                y_pred_proba = model.predict_proba(X_test)[:, 1]
            else:
                y_pred_proba = None

            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)

            metrics = {
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall
            }

            if y_pred_proba is not None:
                auc = roc_auc_score(y_test, y_pred_proba)
                metrics["auc"] = auc

            logger.debug("Model evaluation metrics calculated")

            return metrics

        except Exception as e:
            logger.error("Error during model evaluation: %s", e)
            raise


    def save_metrics(self, metrics):

        try:
            os.makedirs("reports", exist_ok=True)

            with open("reports/metrics.json", "w") as file:
                json.dump(metrics, file, indent=4)

            logger.debug("Metrics saved successfully")

        except Exception as e:
            logger.error("Error saving metrics: %s", e)
            raise


    def run(self, model, X_test, y_test):

        metrics = self.evaluate_model(model, X_test, y_test)

        self.save_metrics(metrics)

        print("Model Metrics:")
        print(metrics)

        return metrics
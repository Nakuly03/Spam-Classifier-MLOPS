import logging

from src.data_ingestion import DataIngestion
from src.pre_processing import DataPreprocessing
from src.feature_engineering import FeatureEngineering
from src.model_training import ModelTraining
from src.model_evaluation import ModelEvaluation


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)


def run_pipeline():

    logging.info("Starting ML Pipeline")

    ingestion = DataIngestion()
    train_path, test_path = ingestion.run()
    logging.info("Data Ingestion Completed")


    preprocessing = DataPreprocessing()
    train_processed, test_processed = preprocessing.run(
        train_path,
        test_path
    )
    logging.info("Data Preprocessing Completed")


    feature_engineering = FeatureEngineering()
    X_train, X_test, y_train, y_test = feature_engineering.run()
    logging.info("Feature Engineering Completed")


    trainer = ModelTraining()
    model = trainer.run(
        X_train,
        y_train
    )
    logging.info("Model Training Completed")



    evaluator = ModelEvaluation()
    metrics = evaluator.run(
        model,
        X_test,
        y_test
    )
    logging.info("Model Evaluation Completed")

    print("\nFinal Model Metrics:")
    print(metrics)


if __name__ == "__main__":
    run_pipeline()
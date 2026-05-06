import os
import sys
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass

from src.exception import CustomException
from src.logger import logging


@dataclass
class DataIngestionConfig:
    raw_data_path: str = os.path.join("artifacts", "data.csv")
    train_data_path: str = os.path.join("artifacts", "train.csv")
    test_data_path: str = os.path.join("artifacts", "test.csv")


class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("Data ingestion started")
        try:
            # ── Read raw dataset ──────────────────────────────────────────────
            df = pd.read_csv("notebooks/data/fake_job_postings.csv")
            logging.info(f"Dataset loaded successfully. Shape: {df.shape}")

            # Drop job_id – not a feature
            if "job_id" in df.columns:
                df.drop(columns=["job_id"], inplace=True)
                logging.info("Dropped column: job_id")

            # ── Save raw file ─────────────────────────────────────────────────
            os.makedirs(
                os.path.dirname(self.ingestion_config.raw_data_path), exist_ok=True
            )
            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)
            logging.info(f"Raw data saved at: {self.ingestion_config.raw_data_path}")

            # ── Stratified train/test split (80/20) ───────────────────────────
            logging.info("Initiating train-test split (stratified on 'fraudulent')")
            train_set, test_set = train_test_split(
                df,
                test_size=0.2,
                random_state=42,
                stratify=df["fraudulent"],
            )

            train_set.to_csv(
                self.ingestion_config.train_data_path, index=False, header=True
            )
            test_set.to_csv(
                self.ingestion_config.test_data_path, index=False, header=True
            )

            logging.info(
                f"Train set: {train_set.shape}  |  Test set: {test_set.shape}"
            )
            logging.info("Data ingestion completed successfully")

            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path,
            )

        except Exception as e:
            raise CustomException(e, sys)


# ── Run the full pipeline when this script is executed directly ───────────────
if __name__ == "__main__":
    from src.components.data_transformation import DataTransformation
    from src.components.model_trainer import ModelTrainer

    # Step 1 – Ingestion
    obj = DataIngestion()
    train_path, test_path = obj.initiate_data_ingestion()

    # Step 2 – Transformation
    data_transformation = DataTransformation()
    train_arr, test_arr, _ = data_transformation.initiate_data_transformation(
        train_path, test_path
    )

    # Step 3 – Model Training
    model_trainer = ModelTrainer()
    f1 = model_trainer.initiate_model_trainer(train_arr, test_arr)
    print(f"\n✅  Best model F1 Score on test set: {f1:.4f}")

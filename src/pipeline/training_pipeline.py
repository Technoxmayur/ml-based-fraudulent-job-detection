import sys
from src.exception import CustomException
from src.logger import logging

from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer


class TrainingPipeline:
    """End-to-end training pipeline: ingest → transform → train."""

    def run_pipeline(self):
        try:
            logging.info("=" * 60)
            logging.info("          TRAINING PIPELINE STARTED")
            logging.info("=" * 60)

            # ── Stage 1: Data Ingestion ────────────────────────────────────────
            logging.info("Stage 1: Data Ingestion")
            data_ingestion = DataIngestion()
            train_path, test_path = data_ingestion.initiate_data_ingestion()

            # ── Stage 2: Data Transformation ──────────────────────────────────
            logging.info("Stage 2: Data Transformation")
            data_transformation = DataTransformation()
            train_arr, test_arr, preprocessor_path = (
                data_transformation.initiate_data_transformation(train_path, test_path)
            )

            # ── Stage 3: Model Training ────────────────────────────────────────
            logging.info("Stage 3: Model Training")
            model_trainer = ModelTrainer()
            f1 = model_trainer.initiate_model_trainer(train_arr, test_arr)

            logging.info("=" * 60)
            logging.info(f"  PIPELINE COMPLETED  |  Best F1: {f1:.4f}")
            logging.info("=" * 60)

            return f1

        except Exception as e:
            raise CustomException(e, sys)


if __name__ == "__main__":
    pipeline = TrainingPipeline()
    score = pipeline.run_pipeline()
    print(f"\n✅  Training complete. Best model F1 Score: {score:.4f}")

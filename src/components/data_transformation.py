import os
import sys
import numpy as np
import pandas as pd
from dataclasses import dataclass
from scipy.sparse import hstack, csr_matrix

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object


@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path: str = os.path.join("artifacts", "preprocessor.pkl")


class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

        # ── Column definitions ─────────────────────────────────────────────────
        self.text_columns = [
            "title",
            "company_profile",
            "description",
            "requirements",
            "benefits",
        ]
        self.categorical_columns = [
            "employment_type",
            "required_experience",
            "required_education",
            "industry",
            "function",
        ]
        self.numerical_columns = [
            "telecommuting",
            "has_company_logo",
            "has_questions",
        ]
        self.target_column = "fraudulent"

    # ── Build sklearn pipelines ────────────────────────────────────────────────
    def get_data_transformer_object(self):
        try:
            logging.info("Building preprocessing pipelines")

            # TF-IDF on concatenated text fields
            # bigrams (1,2), top-5000 features, English stop-words removed
            text_pipeline = Pipeline(
                steps=[
                    (
                        "tfidf",
                        TfidfVectorizer(
                            max_features=5000,
                            stop_words="english",
                            ngram_range=(1, 2),
                            sublinear_tf=True,   # log(1+tf) – helps with freq skew
                        ),
                    )
                ]
            )

            # Categorical: fill missing with most-frequent, then one-hot encode
            cat_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=True)),
                ]
            )

            # Numerical: fill missing with median, then standard-scale
            num_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler(with_mean=False)),  # sparse-safe
                ]
            )

            logging.info(
                f"Text columns    : {self.text_columns}\n"
                f"Categorical cols: {self.categorical_columns}\n"
                f"Numerical cols  : {self.numerical_columns}"
            )

            return text_pipeline, cat_pipeline, num_pipeline

        except Exception as e:
            raise CustomException(e, sys)

    # ── Helper: combine text columns into one string per row ──────────────────
    @staticmethod
    def _combine_text(df: pd.DataFrame, text_columns: list) -> pd.Series:
        for col in text_columns:
            df[col] = df[col].fillna("")
        return df[text_columns].apply(lambda row: " ".join(row.values), axis=1)

    # ── Main transformation method ─────────────────────────────────────────────
    def initiate_data_transformation(self, train_path: str, test_path: str):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info(
                f"Train shape: {train_df.shape}  |  Test shape: {test_df.shape}"
            )
            logging.info("Obtaining preprocessing pipelines")

            text_pipeline, cat_pipeline, num_pipeline = (
                self.get_data_transformer_object()
            )

            # ── Separate features and target ───────────────────────────────────
            y_train = train_df[self.target_column].values
            y_test  = test_df[self.target_column].values

            X_train_df = train_df.drop(columns=[self.target_column])
            X_test_df  = test_df.drop(columns=[self.target_column])

            # ── Combined text feature ──────────────────────────────────────────
            train_text = self._combine_text(X_train_df, self.text_columns)
            test_text  = self._combine_text(X_test_df,  self.text_columns)

            logging.info("Fitting and transforming text features (TF-IDF)")
            X_train_text = text_pipeline.fit_transform(train_text)
            X_test_text  = text_pipeline.transform(test_text)

            # ── Categorical features ───────────────────────────────────────────
            logging.info("Fitting and transforming categorical features")
            X_train_cat = cat_pipeline.fit_transform(
                X_train_df[self.categorical_columns]
            )
            X_test_cat  = cat_pipeline.transform(
                X_test_df[self.categorical_columns]
            )

            # ── Numerical features ─────────────────────────────────────────────
            logging.info("Fitting and transforming numerical features")
            X_train_num = num_pipeline.fit_transform(
                X_train_df[self.numerical_columns]
            )
            X_test_num  = num_pipeline.transform(
                X_test_df[self.numerical_columns]
            )

            # ── Stack all feature blocks (sparse + sparse + sparse) ────────────
            X_train_combined = hstack(
                [X_train_text, X_train_cat, csr_matrix(X_train_num)]
            )
            X_test_combined  = hstack(
                [X_test_text,  X_test_cat,  csr_matrix(X_test_num)]
            )

            logging.info(
                f"Final train feature matrix: {X_train_combined.shape}  "
                f"|  Final test feature matrix: {X_test_combined.shape}"
            )

            # ── Save the full preprocessor bundle ─────────────────────────────
            preprocessor = {
                "text_pipeline":        text_pipeline,
                "cat_pipeline":         cat_pipeline,
                "num_pipeline":         num_pipeline,
                "text_columns":         self.text_columns,
                "categorical_columns":  self.categorical_columns,
                "numerical_columns":    self.numerical_columns,
            }

            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessor,
            )
            logging.info("Preprocessor object saved successfully")

            # Return as tuples (X, y) so model_trainer can unpack easily
            train_arr = (X_train_combined, y_train)
            test_arr  = (X_test_combined,  y_test)

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path,
            )

        except Exception as e:
            raise CustomException(e, sys)

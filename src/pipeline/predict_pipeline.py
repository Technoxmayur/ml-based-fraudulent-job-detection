import os
import sys
import pandas as pd
from scipy.sparse import hstack, csr_matrix

from src.exception import CustomException
from src.logger import logging
from src.utils import load_object


class PredictPipeline:
    """Loads saved model + preprocessor and returns prediction for one record."""

    def __init__(self):
        self.model_path       = os.path.join("artifacts", "model.pkl")
        self.preprocessor_path = os.path.join("artifacts", "preprocessor.pkl")

    def predict(self, features: pd.DataFrame):
        try:
            logging.info("Loading model and preprocessor from artifacts")
            model        = load_object(file_path=self.model_path)
            preprocessor = load_object(file_path=self.preprocessor_path)

            text_pipeline       = preprocessor["text_pipeline"]
            cat_pipeline        = preprocessor["cat_pipeline"]
            num_pipeline        = preprocessor["num_pipeline"]
            text_columns        = preprocessor["text_columns"]
            categorical_columns = preprocessor["categorical_columns"]
            numerical_columns   = preprocessor["numerical_columns"]

            # ── Combine text columns ───────────────────────────────────────────
            for col in text_columns:
                features[col] = features[col].fillna("")
            features["combined_text"] = features[text_columns].apply(
                lambda row: " ".join(row.values), axis=1
            )

            # ── Transform each feature group ───────────────────────────────────
            logging.info("Transforming input features")
            X_text = text_pipeline.transform(features["combined_text"])
            X_cat  = cat_pipeline.transform(features[categorical_columns])
            X_num  = num_pipeline.transform(features[numerical_columns])

            X_combined = hstack([X_text, X_cat, csr_matrix(X_num)])

            # ── Predict ────────────────────────────────────────────────────────
            logging.info("Running prediction")
            preds = model.predict(X_combined)
            logging.info(f"Prediction result: {preds}")
            return preds

        except Exception as e:
            raise CustomException(e, sys)


class CustomData:
    """
    Maps raw HTML form inputs to a pandas DataFrame that the
    preprocessing pipeline understands.
    """

    def __init__(
        self,
        title: str,
        company_profile: str,
        description: str,
        requirements: str,
        benefits: str,
        employment_type: str,
        required_experience: str,
        required_education: str,
        industry: str,
        function: str,
        telecommuting: int,
        has_company_logo: int,
        has_questions: int,
    ):
        self.title               = title
        self.company_profile     = company_profile
        self.description         = description
        self.requirements        = requirements
        self.benefits            = benefits
        self.employment_type     = employment_type
        self.required_experience = required_experience
        self.required_education  = required_education
        self.industry            = industry
        self.function            = function
        self.telecommuting       = telecommuting
        self.has_company_logo    = has_company_logo
        self.has_questions       = has_questions

    def get_data_as_data_frame(self) -> pd.DataFrame:
        try:
            custom_data_input_dict = {
                "title":               [self.title],
                "company_profile":     [self.company_profile],
                "description":         [self.description],
                "requirements":        [self.requirements],
                "benefits":            [self.benefits],
                "employment_type":     [self.employment_type],
                "required_experience": [self.required_experience],
                "required_education":  [self.required_education],
                "industry":            [self.industry],
                "function":            [self.function],
                "telecommuting":       [self.telecommuting],
                "has_company_logo":    [self.has_company_logo],
                "has_questions":       [self.has_questions],
            }
            logging.info("CustomData converted to DataFrame successfully")
            return pd.DataFrame(custom_data_input_dict)

        except Exception as e:
            raise CustomException(e, sys)

import os
import sys
import dill
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.model_selection import GridSearchCV

from src.exception import CustomException
from src.logger import logging


def save_object(file_path, obj):
    """Serializes and saves a Python object to a file using dill."""
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)

        logging.info(f"Object saved successfully at: {file_path}")

    except Exception as e:
        raise CustomException(e, sys)


def load_object(file_path):
    """Loads and deserializes a Python object from a file using dill."""
    try:
        with open(file_path, "rb") as file_obj:
            return dill.load(file_obj)

    except Exception as e:
        raise CustomException(e, sys)


def evaluate_models(X_train, y_train, X_test, y_test, models: dict, param: dict):
    """
    Trains and evaluates multiple classification models.
    Uses GridSearchCV for hyperparameter tuning.
    Returns a report dict of {model_name: f1_score}.
    """
    try:
        report = {}

        for model_name, model in models.items():
            logging.info(f"Training model: {model_name}")
            para = param.get(model_name, {})

            if para:
                logging.info(f"Running GridSearchCV for {model_name}")
                gs = GridSearchCV(model, para, cv=3, scoring="f1", n_jobs=-1)
                gs.fit(X_train, y_train)
                model = gs.best_estimator_
                logging.info(f"Best params for {model_name}: {gs.best_params_}")
            else:
                model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            score = f1_score(y_test, y_pred)

            logging.info(f"{model_name} → F1 Score: {score:.4f}")
            report[model_name] = score

        return report

    except Exception as e:
        raise CustomException(e, sys)

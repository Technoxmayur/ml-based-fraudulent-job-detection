import os
import sys
from dataclasses import dataclass

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import MultinomialNB
from xgboost import XGBClassifier
from sklearn.metrics import (
    f1_score,
    accuracy_score,
    roc_auc_score,
    classification_report,
)

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, evaluate_models


@dataclass
class ModelTrainerConfig:
    trained_model_file_path: str = os.path.join("artifacts", "model.pkl")


class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_arr: tuple, test_arr: tuple):
        try:
            logging.info("Unpacking train and test arrays")
            X_train, y_train = train_arr
            X_test,  y_test  = test_arr

            logging.info(
                f"Training samples: {X_train.shape[0]}  |  "
                f"Test samples: {X_test.shape[0]}"
            )

            # ── Class imbalance weight ─────────────────────────────────────────
            neg = int((y_train == 0).sum())
            pos = int((y_train == 1).sum())
            scale_pos = neg / pos
            logging.info(
                f"Class distribution – Genuine: {neg}  |  Fraudulent: {pos}  "
                f"|  scale_pos_weight: {scale_pos:.2f}"
            )

            # ── Model zoo ─────────────────────────────────────────────────────
            models = {
                "Logistic Regression": LogisticRegression(
                    max_iter=1000, class_weight="balanced", solver="lbfgs"
                ),
                "Naive Bayes": MultinomialNB(),
                "Decision Tree": DecisionTreeClassifier(
                    class_weight="balanced", random_state=42
                ),
                "Random Forest": RandomForestClassifier(
                    class_weight="balanced", n_estimators=100, random_state=42
                ),
                "Gradient Boosting": GradientBoostingClassifier(
                    n_estimators=100, random_state=42
                ),
                "XGBoost": XGBClassifier(
                    scale_pos_weight=scale_pos,
                    eval_metric="logloss",
                    random_state=42,
                    n_estimators=100,
                ),
            }

            # ── Hyperparameter search space ────────────────────────────────────
            params = {
                "Logistic Regression": {"C": [0.1, 1.0, 10.0]},
                "Naive Bayes":         {"alpha": [0.1, 0.5, 1.0]},
                "Decision Tree":       {"max_depth": [5, 10, 20, None]},
                "Random Forest":       {"n_estimators": [100, 200]},
                "Gradient Boosting":   {
                    "n_estimators": [100, 200],
                    "learning_rate": [0.05, 0.1],
                },
                "XGBoost": {
                    "n_estimators": [100, 200],
                    "learning_rate": [0.05, 0.1],
                },
            }

            # ── Evaluate all models ────────────────────────────────────────────
            logging.info("Starting model evaluation with GridSearchCV")
            model_report: dict = evaluate_models(
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
                models=models,
                param=params,
            )

            logging.info("Model F1 Scores:")
            for name, score in sorted(model_report.items(), key=lambda x: -x[1]):
                logging.info(f"  {name:25s} → F1: {score:.4f}")

            # ── Pick best model ────────────────────────────────────────────────
            best_model_name  = max(model_report, key=model_report.get)
            best_model_score = model_report[best_model_name]
            best_model       = models[best_model_name]

            logging.info(
                f"Best model: {best_model_name}  |  F1 Score: {best_model_score:.4f}"
            )

            if best_model_score < 0.60:
                raise CustomException(
                    "No acceptable model found (F1 < 0.60). "
                    "Check data quality and class balance.",
                    sys,
                )

            # ── Detailed evaluation on test set ────────────────────────────────
            y_pred = best_model.predict(X_test)
            acc    = accuracy_score(y_test, y_pred)
            f1     = f1_score(y_test, y_pred)
            roc    = roc_auc_score(y_test, y_pred)

            logging.info(
                f"\n{'='*50}\n"
                f"  Best Model   : {best_model_name}\n"
                f"  Accuracy     : {acc:.4f}\n"
                f"  F1 Score     : {f1:.4f}\n"
                f"  ROC-AUC      : {roc:.4f}\n"
                f"{'='*50}"
            )
            logging.info(
                f"\nClassification Report:\n"
                f"{classification_report(y_test, y_pred, target_names=['Genuine','Fraudulent'])}"
            )

            # ── Save best model ────────────────────────────────────────────────
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model,
            )
            logging.info(
                f"Best model saved at: {self.model_trainer_config.trained_model_file_path}"
            )

            return f1

        except Exception as e:
            raise CustomException(e, sys)

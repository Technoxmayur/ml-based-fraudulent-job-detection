from flask import Flask, request, render_template
import numpy as np
import pandas as pd

from src.pipeline.predict_pipeline import CustomData, PredictPipeline

application = Flask(__name__)
app = application


# ── Home page ─────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html")


# ── Prediction endpoint ────────────────────────────────────────────────────────
@app.route("/predictdata", methods=["GET", "POST"])
def predict_datapoint():
    if request.method == "GET":
        return render_template("home.html")

    # POST – collect form data, run prediction
    data = CustomData(
        title=request.form.get("title", ""),
        company_profile=request.form.get("company_profile", ""),
        description=request.form.get("description", ""),
        requirements=request.form.get("requirements", ""),
        benefits=request.form.get("benefits", ""),
        employment_type=request.form.get("employment_type", ""),
        required_experience=request.form.get("required_experience", ""),
        required_education=request.form.get("required_education", ""),
        industry=request.form.get("industry", ""),
        function=request.form.get("function", ""),
        telecommuting=1 if request.form.get("telecommuting") == "1" else 0,
        has_company_logo=1 if request.form.get("has_company_logo") == "1" else 0,
        has_questions=1 if request.form.get("has_questions") == "1" else 0,
    )

    pred_df = data.get_data_as_data_frame()
    print("[INFO] Input DataFrame:\n", pred_df)
    print("[INFO] Running prediction...")

    predict_pipeline = PredictPipeline()
    results = predict_pipeline.predict(pred_df)

    print("[INFO] Prediction complete:", results)

    label = int(results[0])
    if label == 1:
        result_text = "⚠️ FRAUDULENT – This job posting appears to be fake!"
        result_class = "danger"
    else:
        result_text = "✅ GENUINE – This job posting appears to be legitimate."
        result_class = "success"

    return render_template("home.html", results=result_text, result_class=result_class)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)

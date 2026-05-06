#  Fraud Job Detection – NLP Machine Learning Project

### Software and Tools Required
1. [GitHub Account](https://github.com)
2. [VS Code](https://code.visualstudio.com)
3. Python 3.10+
4. Kaggle Account (to download dataset)

---

##  Overview

This project detects **fraudulent job postings** on online platforms using *Natural Language Processing (NLP)* and Machine Learning. It analyses job descriptions, company profiles, requirements, benefits, and posting metadata to classify each listing as **genuine** or **fraudulent**.

The project follows a **modular ML pipeline architecture** with custom logging, exception handling, and a Flask web application for live predictions.

---

##  Problem Statement

> The rise of fraudulent job postings on online portals puts applicants at financial and personal risk. Manual verification is slow and unscalable. This project builds an automated ML classifier that analyses textual and metadata patterns to distinguish genuine job postings from fraudulent ones.

---

##  Project Structure

```
Fraud-Job-Detection/
│
├── artifacts/                    
│   ├── data.csv
│   ├── train.csv
│   ├── test.csv
│   ├── preprocessor.pkl
│   └── model.pkl
│
├── logs/                         
│
├── notebooks/                    
│   ├── data/
│   │    └── fake_job_postings.csv   
│   ├── 1. EDA_FRAUD_JOB_DETECTION.ipynb
│   └── 2. MODEL_TRAINING.ipynb
│
├── src/                         
│   ├── __init__.py
│   ├── exception.py              
│   ├── logger.py                 
│   ├── utils.py                  
│   │
│   ├── components/
│   │    ├── __init__.py
│   │    ├── data_ingestion.py    
│   │    ├── data_transformation.py  
│   │    └── model_trainer.py   
│   │
│   └── pipeline/
│        ├── __init__.py
│        ├── predict_pipeline.py 
│        └── training_pipeline.py 
│
├── templates/
│   ├── index.html               
│   └── home.html               
│
├── app.py                       
├── requirements.txt
├── setup.py
├── .gitignore
└── README.md
```

---

##  Tech Stack

| Category | Tools |
|---|---|
| Language | Python 3.10 |
| NLP | TF-IDF (sklearn), NLTK |
| ML Models | Logistic Regression, Naive Bayes, Decision Tree, Random Forest, Gradient Boosting, XGBoost |
| Preprocessing | scikit-learn Pipelines |
| Serialization | dill |
| Web Framework | Flask |
| Visualization | Matplotlib, Seaborn, WordCloud |
| Data | Pandas, NumPy, SciPy (sparse matrices) |

---

##  Features Used

### Text Features (combined → TF-IDF with bigrams)
- `title`
- `company_profile`
- `description`
- `requirements`
- `benefits`

### Categorical Features (OneHotEncoded)
- `employment_type`
- `required_experience`
- `required_education`
- `industry`
- `function`

### Binary / Numerical Features
- `telecommuting` (0/1)
- `has_company_logo` (0/1)
- `has_questions` (0/1)

### Target
- `fraudulent` (0 = genuine, 1 = fraudulent)

---

## 🔄 ML Pipeline Workflow

### Stage 1 – Data Ingestion
- Reads `fake_job_postings.csv` from `notebooks/data/`
- Drops `job_id`
- Stratified 80/20 train-test split
- Saves `data.csv`, `train.csv`, `test.csv` to `artifacts/`

### Stage 2 – Data Transformation
- Fills NaN in text columns with empty string
- Concatenates all text columns into `combined_text`
- Applies TF-IDF (5000 features, bigrams, sublinear TF)
- Imputes + OneHot-encodes categorical features
- Imputes + StandardScales numerical features
- Stacks all features into a sparse matrix
- Saves `preprocessor.pkl` to `artifacts/`

### Stage 3 – Model Training
- Trains 6 classifiers with `class_weight='balanced'`
- Uses GridSearchCV for hyperparameter tuning
- Evaluates on F1 Score (best for imbalanced data)
- Saves best model to `artifacts/model.pkl`

---

##  How to Run

### Step 1: Clone the repository
```bash
git clone <your-repo-link>
cd Fraud-Job-Detection
```

### Step 2: Download the dataset
Download `fake_job_postings.csv` from Kaggle and place it in:
```
notebooks/data/fake_job_postings.csv
```

### Step 3: Create virtual environment
```bash
python -m venv venv
venv\Scripts\activate   
```

### Step 4: Install dependencies
```bash
pip install -r requirements.txt
```

### Step 5: Run the training pipeline
```bash
python -m src.components.data_ingestion
```
This runs all three stages automatically and saves model + preprocessor to `artifacts/`.

### Step 6: Start the Flask app
```bash
python app.py
```
Open your browser at `http://localhost:5000`

---

##  Sample Output

After running the pipeline:
```
artifacts/
 ├── data.csv
 ├── train.csv
 ├── test.csv
 ├── preprocessor.pkl
 └── model.pkl

logs/
 └── 05_15_2024_10_30_00/
      └── 05_15_2024_10_30_00.log
```

---

##  Key Highlights

- Modular NLP + ML pipeline architecture
- Handles severe class imbalance (~4.8% fraud rate)
- Custom logging (timestamped) and exception handling with file/line details
- Sparse matrix operations for efficient TF-IDF + other feature combination
- 6 models compared; best selected automatically by F1 score
- Flask web app for real-time fraud detection

---

##  Common Issues & Fixes

| Issue | Solution |
|---|---|
| `FileNotFoundError` for CSV | Place dataset in `notebooks/data/` |
| Import errors | Run as `python -m src.components.data_ingestion` |
| Kernel crash in notebook | Install `ipykernel` in venv |
| Model not saving | Check `artifacts/` directory permissions |
| `sparse_output` error | Upgrade scikit-learn ≥ 1.2 |

---

##  Future Improvements

- BERT / Sentence Transformers for richer text embeddings
- Add location-based fraud patterns
- Real-time scraping + detection pipeline
- Docker containerisation
- CI/CD with GitHub Actions
- Model monitoring with Evidently AI

---

##  Team

| Name | Roll No. | Contribution | % |
|---|---|---|---|
| Member 1 | - | Data Ingestion, Logger, Exception | 25% |
| Member 2 | - | EDA Notebook | 25% |
| Member 3 | - | Data Transformation, Preprocessing | 25% |
| Member 4 | - | Model Training, Flask App | 25% |

---

##  Acknowledgements

- Kaggle Dataset – Fake Job Posting Prediction
- Scikit-learn documentation
- XGBoost documentation

---

##  Contact

Feel free to connect for collaboration or queries.

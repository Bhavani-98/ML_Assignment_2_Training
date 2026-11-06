# Spam E-mail Classification — ML Assignment 2

A comparison of five supervised classifiers for detecting spam e-mail, with an
interactive Streamlit app for scoring test data against the trained models.

**Student:** Bhavani S
**BITS ID:** 2026AJ05027
**Course:** M.Tech (AIML) — Machine Learning — Assignment 2

---

## a. Problem Statement

Given 57 numeric statistics extracted from an e-mail (frequencies of specific words
and characters, plus runs of capital letters), decide whether the message is **spam
(1)** or **legitimate / ham (0)**. This is a **binary classification** task. Five
classifiers are trained on the same corpus, compared across six metrics, and exposed
through a Streamlit web app that loads the saved models and scores an uploaded test
CSV **without any retraining**.

## b. Dataset Description

- **Source:** UCI Machine Learning Repository — *Spambase* (ID 94),
  https://archive.ics.uci.edu/dataset/94/spambase
- **Instances:** 4,601 e-mails  (≥ 500 required ✔)
- **Features:** 57 numeric attributes  (≥ 12 required ✔)
  - 48 `word_freq_*` — percentage of words matching a given term
  - 6 `char_freq_*` — percentage of characters matching `; ( [ ! $ #`
  - 3 capital-run statistics — average, longest and total runs of capital letters
- **Target:** `is_spam` — 1 = spam (1,813 messages, 39.4%), 0 = ham (2,788 messages)
- **Missing values:** none

**Pre-processing.** Each model is wrapped in a scikit-learn `Pipeline` of
`StandardScaler → estimator`, so the features are standardised using statistics learned
on the training fold only, and that scaler is serialised *together with* the estimator.
At serving time the app simply calls `pipeline.predict`, which guarantees the uploaded
test data is transformed exactly as during training. The data is split 75/25 with
stratification to preserve the ham:spam ratio.

## c. GitHub Repository Link

`https://github.com/Bhavani-98/ML_Assignment_2_Training`

## d. Models Used

All five classifiers are trained on the same 75% training partition and scored on the
untouched 25% hold-out (1,151 messages). Being a binary task, AUC uses the positive-class
probability and Precision/Recall/F1 are for the spam class.

### Comparison Table

| ML Model Name           | Accuracy | AUC    | Precision | Recall | F1     | MCC    |
|-------------------------|----------|--------|-----------|--------|--------|--------|
| Logistic Regression     | 0.9288   | 0.9745 | 0.9227    | 0.8943 | 0.9083 | 0.8503 |
| Decision Tree           | 0.9166   | 0.9627 | 0.9050    | 0.8811 | 0.8929 | 0.8248 |
| KNN                     | 0.9201   | 0.9704 | 0.9022    | 0.8943 | 0.8982 | 0.8324 |
| Naive Bayes             | 0.8149   | 0.9164 | 0.6947    | 0.9471 | 0.8015 | 0.6626 |
| Support Vector Machine  | 0.9357   | 0.9799 | 0.9318    | 0.9031 | 0.9172 | 0.8650 |

### Observations

| ML Model Name | Observation about model performance |
|---|---|
| **Logistic Regression** | A strong, well-balanced linear baseline (F1 0.908, AUC 0.975). The log-frequency features are largely linearly separable, so a linear boundary already classifies most messages correctly with balanced precision and recall. |
| **Decision Tree** | Slightly weaker (F1 0.893). A single depth-12 tree captures obvious keyword rules (e.g. frequent `!`, `$`, `remove`) but misses subtler feature combinations and is more prone to variance than the linear/margin models. |
| **KNN** | Competitive after scaling (F1 0.898, k = 11, distance-weighted). It relies on neighbourhood structure in the 57-dimensional space; standardisation inside the pipeline is what makes it work. |
| **Naive Bayes** | Distinct behaviour: the **highest recall (0.947)** — it catches the most spam — but the **lowest precision (0.695)**, because its feature-independence assumption is violated by strongly correlated word frequencies, so it over-flags legitimate mail. Lowest overall accuracy and MCC. |
| **Support Vector Machine** | **Best overall** — top Accuracy, AUC, F1 and MCC. The RBF-kernel maximum-margin boundary models the high-dimensional, non-linear separation between spam and ham most effectively. |
| **Overall, Winner for your dataset?** | **Support Vector Machine** (F1 = 0.9172, MCC = 0.8650). For a practical spam filter, where wrongly flagging real mail (a false positive) is costly, SVM and Logistic Regression are preferable to Naive Bayes despite NB's high recall, because they keep precision high. |

## Repository Layout

```
project/
├── app.py                         # Streamlit app (loads pipelines, no retraining)
├── requirements.txt               # Pinned dependencies
├── README.md                      # This file
├── sample_test.csv                # 25% hold-out split for uploading into the app
├── src/
│   └── build_classifiers.py       # Trains the five pipelines, writes the model/ files
├── notebooks/
│   └── spam_classification.ipynb  # EDA + training + evaluation (executed, with outputs)
└── model/
    ├── pipeline_logistic_regression.pkl
    ├── pipeline_decision_tree.pkl
    ├── pipeline_k_nearest_neighbors.pkl
    ├── pipeline_naive_bayes.pkl
    ├── pipeline_support_vector_machine.pkl
    ├── schema.pkl                 # feature list, label, class names, pipeline map
    └── scoreboard.csv             # saved comparison table
```

Each `pipeline_*.pkl` is a complete `StandardScaler → classifier` pipeline, so no
separate scaler file is needed.

## Live Streamlit App

`https://<your-app>.streamlit.app`   *(fill in after deploying)*

**App features**
- **Upload** a labelled test CSV.
- **Single model** tab — pick any classifier from a dropdown and see its six metrics.
- **All models** tab — a side-by-side comparison table (best value highlighted) and a bar chart.
- **Detailed report** tab — confusion matrix and a full classification report.

The app loads the pre-fitted pipelines from `model/` and scores the uploaded data;
it never re-trains on the uploaded rows.

## Run Locally

```bash
pip install -r requirements.txt
python src/build_classifiers.py      # (optional) rebuild the model/ pipelines
streamlit run app.py                 # then upload sample_test.csv
```

> The repository already ships the trained pipelines in `model/` and the
> `sample_test.csv` hold-out split, so you can run the app and upload the CSV
> directly without re-running `build_classifiers.py` first.

## Deploy on Streamlit Community Cloud

1. Push this folder to a public GitHub repository.
2. Open https://streamlit.io/cloud and sign in with GitHub.
3. **New app** → pick the repo → branch `main` → main file `app.py` → **Deploy**.
4. When the app is live, upload `sample_test.csv` to view the results.

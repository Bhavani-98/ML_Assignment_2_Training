"""
Spam Detection Explorer - Streamlit front-end.

Loads the five pre-fitted scikit-learn pipelines and scores a user-supplied test
CSV. The pipelines already contain their scaler, so this layer never touches the
preprocessing or re-fits anything - it simply predicts and reports metrics.

Author: Bhavani S (BITS ID: 2026AJ05027)
"""

import pathlib
import pickle

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from sklearn import metrics

ART = pathlib.Path(__file__).resolve().parent / "artifacts"

st.set_page_config(page_title="Spam Detection Explorer", page_icon="📧", layout="wide")


@st.cache_resource
def bootstrap():
    """Load schema + all pipelines once per session."""
    with open(ART / "schema.pkl", "rb") as fh:
        schema = pickle.load(fh)
    pipelines = {}
    for name, fname in schema["pipelines"].items():
        with open(ART / fname, "rb") as fh:
            pipelines[name] = pickle.load(fh)
    return schema, pipelines


def score(pipe, X, y):
    """Return the six evaluation numbers for a binary classifier."""
    pred = pipe.predict(X)
    try:
        prob = pipe.predict_proba(X)[:, 1]
        auc = metrics.roc_auc_score(y, prob)
    except Exception:
        auc = np.nan
    return {
        "Accuracy": metrics.accuracy_score(y, pred),
        "AUC": auc,
        "Precision": metrics.precision_score(y, pred, zero_division=0),
        "Recall": metrics.recall_score(y, pred, zero_division=0),
        "F1": metrics.f1_score(y, pred, zero_division=0),
        "MCC": metrics.matthews_corrcoef(y, pred),
    }, pred


# ----------------------------------------------------------------------------
# Header + sidebar
# ----------------------------------------------------------------------------
st.title("📧 Spam Detection Explorer")
st.write(
    "Benchmark five classifiers trained on the UCI **Spambase** corpus. "
    "Upload a labelled test CSV and the app will report metrics for any model "
    "you pick - **no training happens in this app**."
)

try:
    SCHEMA, PIPES = bootstrap()
except FileNotFoundError:
    st.error("Artifacts missing. Please run `python src/build_classifiers.py` first.")
    st.stop()

FEATURES = SCHEMA["features"]
LABEL = SCHEMA["label"]
CLASS_LABELS = SCHEMA["class_labels"]

with st.sidebar:
    st.subheader("Project")
    st.markdown(
        "- **Task:** binary spam classification\n"
        "- **Dataset:** UCI Spambase (4,601 e-mails, 57 features)\n"
        "- **Models:** LR, Decision Tree, KNN, Naive Bayes, SVM"
    )
    st.divider()
    st.subheader("Submission")
    st.markdown("**Student:** Bhavani S\n\n**ID:** 2026AJ05027")
    st.divider()
    st.caption("Upload `sample_test.csv` from the repo, or any CSV with the 57 "
               "Spambase feature columns plus the `is_spam` label.")

# ----------------------------------------------------------------------------
# Upload (shared across tabs)
# ----------------------------------------------------------------------------
upload = st.file_uploader("Upload labelled test data (CSV)", type="csv")

if upload is None:
    st.info("⬆️  Upload a CSV to begin. The repository ships with `sample_test.csv`.")
    st.stop()

raw = pd.read_csv(upload)
gap = [c for c in FEATURES if c not in raw.columns]
if gap:
    st.error(f"CSV is missing {len(gap)} required feature column(s), e.g. {gap[:5]}")
    st.stop()
if LABEL not in raw.columns:
    st.error(f"CSV must include the label column '{LABEL}'.")
    st.stop()

X_in = raw[FEATURES]
y_in = raw[LABEL].astype(int).to_numpy()
st.success(f"Loaded {len(raw):,} rows with {len(FEATURES)} features and the '{LABEL}' label.")

tab_single, tab_all, tab_detail = st.tabs(
    ["🔎 Single model", "📊 All models", "🧩 Detailed report"]
)

# ----------------------------------------------------------------------------
# Tab 1 - single model (dropdown + metrics)
# ----------------------------------------------------------------------------
with tab_single:
    pick = st.selectbox("Select a classifier", list(PIPES.keys()))
    pipe = PIPES[pick]
    res, pred = score(pipe, X_in, y_in)

    st.markdown(f"#### Metrics for **{pick}**")
    left, right = st.columns(2)
    with left:
        st.metric("Accuracy", f"{res['Accuracy']:.3f}")
        st.metric("Precision", f"{res['Precision']:.3f}")
        st.metric("F1 score", f"{res['F1']:.3f}")
    with right:
        st.metric("AUC", "-" if np.isnan(res["AUC"]) else f"{res['AUC']:.3f}")
        st.metric("Recall", f"{res['Recall']:.3f}")
        st.metric("MCC", f"{res['MCC']:.3f}")

# ----------------------------------------------------------------------------
# Tab 2 - all models comparison
# ----------------------------------------------------------------------------
with tab_all:
    st.markdown("#### Side-by-side comparison on the uploaded data")
    table = {name: score(p, X_in, y_in)[0] for name, p in PIPES.items()}
    board = pd.DataFrame(table).T[["Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"]]
    st.dataframe(board.style.format("{:.3f}").highlight_max(axis=0, color="#cdeccd"),
                 use_container_width=True)

    fig, ax = plt.subplots(figsize=(9, 4))
    board[["Accuracy", "F1", "MCC"]].plot.bar(ax=ax)
    ax.set_ylim(0.5, 1.0)
    ax.set_ylabel("score")
    ax.set_title("Accuracy / F1 / MCC by model")
    ax.set_xticklabels(board.index, rotation=20, ha="right")
    st.pyplot(fig)

# ----------------------------------------------------------------------------
# Tab 3 - confusion matrix + classification report
# ----------------------------------------------------------------------------
with tab_detail:
    pick2 = st.selectbox("Classifier for detailed breakdown", list(PIPES.keys()), key="detail")
    pipe2 = PIPES[pick2]
    pred2 = pipe2.predict(X_in)

    names = [CLASS_LABELS["0"], CLASS_LABELS["1"]]
    cm = metrics.confusion_matrix(y_in, pred2)

    colL, colR = st.columns([1, 1])
    with colL:
        st.markdown("**Confusion matrix**")
        fig2, ax2 = plt.subplots(figsize=(4.5, 4))
        im = ax2.imshow(cm, cmap="Purples")
        ax2.set_xticks([0, 1], names)
        ax2.set_yticks([0, 1], names)
        ax2.set_xlabel("Predicted")
        ax2.set_ylabel("Actual")
        for (i, j), v in np.ndenumerate(cm):
            ax2.text(j, i, str(v), ha="center", va="center",
                     color="white" if v > cm.max() / 2 else "black", fontsize=12)
        fig2.colorbar(im, ax=ax2, fraction=0.046)
        st.pyplot(fig2)
    with colR:
        st.markdown("**Classification report**")
        rep = metrics.classification_report(
            y_in, pred2, target_names=names, zero_division=0, output_dict=True
        )
        st.dataframe(pd.DataFrame(rep).T.round(3), use_container_width=True)

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

ART = pathlib.Path(__file__).resolve().parent / "model"
SAMPLE_CSV = pathlib.Path(__file__).resolve().parent / "sample_test.csv"

st.set_page_config(page_title="Spam Detection Explorer", page_icon="📧", layout="wide")


@st.cache_data
def load_sample_bytes():
    """Read the bundled sample_test.csv as raw bytes (for the download button)."""
    if SAMPLE_CSV.exists():
        return SAMPLE_CSV.read_bytes()
    return None


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
    st.error("Saved models missing. Please run `python src/build_classifiers.py` first.")
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
    st.caption("Upload any CSV with the 57 Spambase feature columns plus the "
               "`is_spam` label, or use the bundled `sample_test.csv` below.")
    _sample_bytes = load_sample_bytes()
    if _sample_bytes is not None:
        st.download_button(
            "⬇️ Download sample_test.csv",
            data=_sample_bytes,
            file_name="sample_test.csv",
            mime="text/csv",
            use_container_width=True,
        )

# ----------------------------------------------------------------------------
# Data source (shared across tabs): bundled sample OR user upload
# ----------------------------------------------------------------------------
sample_available = SAMPLE_CSV.exists()

col_u, col_s = st.columns([3, 2])
with col_u:
    upload = st.file_uploader("Upload labelled test data (CSV)", type="csv")
with col_s:
    st.write("")  # spacer to align with the uploader label
    use_sample = st.checkbox(
        "Use bundled sample_test.csv",
        value=False,
        disabled=not sample_available,
        help="Score the hold-out test set that ships with the repository - no upload needed.",
    )

if use_sample and sample_available:
    raw = pd.read_csv(SAMPLE_CSV)
    st.caption("Scoring the bundled **sample_test.csv** (repository hold-out split).")
elif upload is not None:
    raw = pd.read_csv(upload)
else:
    st.info("⬆️  Upload a CSV to begin, or tick **Use bundled sample_test.csv** "
            "to score the test set shipped with the repository.")
    st.stop()

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

# ----------------------------------------------------------------------------
# Section 1 - single model (dropdown + metrics)
# ----------------------------------------------------------------------------
st.divider()
st.subheader("🔎 Single model")
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
# Section 2 - all models comparison
# ----------------------------------------------------------------------------
st.divider()
st.subheader("📊 All models")
st.markdown("#### Side-by-side comparison on the selected data")
table = {name: score(p, X_in, y_in)[0] for name, p in PIPES.items()}
board = pd.DataFrame(table).T[["Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"]]
st.dataframe(board.style.format("{:.3f}").highlight_max(axis=0, color="#cdeccd"),
             use_container_width=True)
st.caption("The best value in each metric column is highlighted in green.")

fig, ax = plt.subplots(figsize=(9, 4))
board[["Accuracy", "F1", "MCC"]].plot.bar(ax=ax)
ax.set_ylim(0.5, 1.0)
ax.set_ylabel("score")
ax.set_title("Accuracy / F1 / MCC by model")
ax.set_xticklabels(board.index, rotation=20, ha="right")
st.pyplot(fig)

# ----------------------------------------------------------------------------
# Section 3 - confusion matrix + classification report
# ----------------------------------------------------------------------------
st.divider()
st.subheader("🧩 Detailed report")
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

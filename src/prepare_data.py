"""
Prepare the raw Spambase dataset for training.

The raw UCI Spambase corpus is intentionally NOT committed to the repository
(see .gitignore -> _spam/). This script downloads it from the UCI repository and
writes it to ``_spam/spambase_full.csv`` with the exact header that
``build_classifiers.py`` expects (57 feature columns + the ``is_spam`` label).

Run this once on a fresh machine (e.g. the VM) before re-training:

    python src/prepare_data.py
    python src/build_classifiers.py

Author: Bhavani S (BITS ID: 2026AJ05027)
"""

from __future__ import annotations

import pathlib
import sys
import urllib.request

PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "_spam"
OUT_CSV = RAW_DIR / "spambase_full.csv"

# Canonical UCI download for the Spambase data (headerless, comma-separated:
# 57 numeric features followed by the 0/1 class label, in this exact order).
DATA_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/spambase/spambase.data"

# Exact column order and names expected by the training pipeline / saved schema.
FEATURE_COLUMNS = [
    "word_freq_make", "word_freq_address", "word_freq_all", "word_freq_3d",
    "word_freq_our", "word_freq_over", "word_freq_remove", "word_freq_internet",
    "word_freq_order", "word_freq_mail", "word_freq_receive", "word_freq_will",
    "word_freq_people", "word_freq_report", "word_freq_addresses", "word_freq_free",
    "word_freq_business", "word_freq_email", "word_freq_you", "word_freq_credit",
    "word_freq_your", "word_freq_font", "word_freq_000", "word_freq_money",
    "word_freq_hp", "word_freq_hpl", "word_freq_george", "word_freq_650",
    "word_freq_lab", "word_freq_labs", "word_freq_telnet", "word_freq_857",
    "word_freq_data", "word_freq_415", "word_freq_85", "word_freq_technology",
    "word_freq_1999", "word_freq_parts", "word_freq_pm", "word_freq_direct",
    "word_freq_cs", "word_freq_meeting", "word_freq_original", "word_freq_project",
    "word_freq_re", "word_freq_edu", "word_freq_table", "word_freq_conference",
    "char_freq_;", "char_freq_(", "char_freq_[", "char_freq_!", "char_freq_$",
    "char_freq_#", "capital_run_length_average", "capital_run_length_longest",
    "capital_run_length_total",
]
LABEL = "is_spam"
EXPECTED_FEATURES = 57


def download_raw() -> str:
    """Download the raw headerless Spambase data and return it as text."""
    print(f"Downloading Spambase data from:\n  {DATA_URL}")
    try:
        with urllib.request.urlopen(DATA_URL, timeout=60) as resp:
            return resp.read().decode("utf-8")
    except Exception as exc:  # pragma: no cover - network/environment dependent
        print(
            "\nERROR: could not download the dataset automatically.\n"
            "On a restricted network, manually download 'spambase.data' from\n"
            "  https://archive.ics.uci.edu/dataset/94/spambase\n"
            f"and place it at: {RAW_DIR / 'spambase.data'}\n"
            "then re-run this script (it will use the local copy).",
            file=sys.stderr,
        )
        raise SystemExit(1) from exc


def load_raw_text() -> str:
    """Prefer a local spambase.data if present; otherwise download it."""
    local = RAW_DIR / "spambase.data"
    if local.exists():
        print(f"Using local raw file: {local}")
        return local.read_text(encoding="utf-8")
    return download_raw()


def main() -> None:
    header = FEATURE_COLUMNS + [LABEL]
    if len(FEATURE_COLUMNS) != EXPECTED_FEATURES:
        raise SystemExit(
            f"Header definition error: expected {EXPECTED_FEATURES} features, "
            f"got {len(FEATURE_COLUMNS)}"
        )

    raw_text = load_raw_text()
    rows = [ln for ln in raw_text.splitlines() if ln.strip()]

    # Validate the first data row has the right number of columns.
    first_cols = rows[0].split(",")
    if len(first_cols) != len(header):
        raise SystemExit(
            f"Raw data has {len(first_cols)} columns but the header expects "
            f"{len(header)}. The source format may have changed."
        )

    RAW_DIR.mkdir(exist_ok=True)
    with open(OUT_CSV, "w", encoding="utf-8", newline="") as fh:
        fh.write(",".join(header) + "\n")
        for line in rows:
            fh.write(line + "\n")

    print(f"Wrote {OUT_CSV}  ({len(rows)} data rows, {len(header)} columns)")
    print("You can now run:  python src/build_classifiers.py")


if __name__ == "__main__":
    main()

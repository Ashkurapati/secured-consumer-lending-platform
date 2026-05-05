"""
train_model.py
--------------
Step 5 of the Stepping Stones ML pipeline.

What this script does:
  1. Loads train.csv and test.csv (output of split_data.py)
  2. Trains a Logistic Regression baseline
  3. Trains a LightGBM model (production candidate)
  4. Evaluates both: AUC-ROC, KS statistic, classification report
  5. Saves the best model as model.pkl

Run on Darwin HPC:
    python train_model.py

Dependencies:
    pip install lightgbm==4.3.0 scikit-learn==1.3.0 pandas==2.0.3 numpy==1.24.0 joblib==1.3.0
"""

import pandas as pd
import numpy as np
import joblib
import os
import time

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score,
    classification_report,
    roc_curve,
)
from lightgbm import LGBMClassifier

TARGET = "default"
SEED   = 42

# ─────────────────────────────────────────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────────────────────────────────────────

print("=" * 60)
print("  Stepping Stones -- Model Training")
print("=" * 60)

print("\n[1/5] Loading train.csv and test.csv ...")

train_df = pd.read_csv("train.csv", low_memory=False)
test_df  = pd.read_csv("test.csv",  low_memory=False)

print(f"      Train : {len(train_df):,} rows x {len(train_df.columns)} columns")
print(f"      Test  : {len(test_df):,} rows x {len(test_df.columns)} columns")

# ─────────────────────────────────────────────────────────────────────────────
# SEPARATE FEATURES AND TARGET
# ─────────────────────────────────────────────────────────────────────────────

print("\n[2/5] Separating features and target ...")

X_train = train_df.drop(columns=[TARGET])
y_train = train_df[TARGET]

X_test  = test_df.drop(columns=[TARGET])
y_test  = test_df[TARGET]

print(f"      Features : {X_train.shape[1]} columns")
print(f"      Train default rate : {y_train.mean():.2%}")
print(f"      Test  default rate : {y_test.mean():.2%}")

# ─────────────────────────────────────────────────────────────────────────────
# HELPER: KS STATISTIC
# ─────────────────────────────────────────────────────────────────────────────

def ks_statistic(y_true, y_prob):
    """
    KS (Kolmogorov-Smirnov) statistic — standard credit risk metric.
    Measures the maximum separation between the cumulative
    distributions of predicted probabilities for defaulters vs non-defaulters.
    A good credit model targets KS > 0.40.
    """
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    return round(float(np.max(tpr - fpr)), 4)


# ─────────────────────────────────────────────────────────────────────────────
# MODEL 1: LOGISTIC REGRESSION (BASELINE)
# ─────────────────────────────────────────────────────────────────────────────

print("\n[3/5] Training Logistic Regression (baseline) ...")
print("      This may take 3-5 minutes ...")

t0 = time.time()

lr_model = LogisticRegression(
    class_weight="balanced",   # handles 78/22 class imbalance
    max_iter=2000,             # increased to ensure convergence on large data
    solver="saga",             # best solver for large datasets
    random_state=SEED,
    n_jobs=-1,                 # use all CPU cores on Darwin
)

lr_model.fit(X_train, y_train)
lr_time = round(time.time() - t0, 1)

lr_probs  = lr_model.predict_proba(X_test)[:, 1]
lr_auc    = round(roc_auc_score(y_test, lr_probs), 4)
lr_ks     = ks_statistic(y_test, lr_probs)

print(f"\n      Logistic Regression Results")
print(f"      Training time : {lr_time}s")
print(f"      AUC-ROC       : {lr_auc}")
print(f"      KS Statistic  : {lr_ks}")
print(f"\n      Classification Report:")
print(classification_report(y_test, lr_model.predict(X_test),
                             target_names=["Paid Off", "Defaulted"]))

# ─────────────────────────────────────────────────────────────────────────────
# MODEL 2: LIGHTGBM (PRODUCTION CANDIDATE)
# ─────────────────────────────────────────────────────────────────────────────

print("\n[4/5] Training LightGBM ...")
print("      This may take 5-10 minutes ...")

t0 = time.time()

lgbm_model = LGBMClassifier(
    n_estimators=1000,         # number of trees
    learning_rate=0.05,        # step size — lower = more trees needed but better generalisation
    num_leaves=63,             # controls complexity — 63 is good for tabular credit data
    min_child_samples=100,     # minimum rows per leaf — prevents overfitting on rare cases
    class_weight="balanced",   # handles 78/22 class imbalance
    random_state=SEED,
    n_jobs=-1,                 # use all CPU cores on Darwin
    verbose=-1,                # suppress per-tree output
    device="cpu",              # GPU build not available on this cluster
)

lgbm_model.fit(
    X_train, y_train,
    eval_set=[(X_test, y_test)],
    callbacks=[],
)
lgbm_time = round(time.time() - t0, 1)

lgbm_probs = lgbm_model.predict_proba(X_test)[:, 1]
lgbm_auc   = round(roc_auc_score(y_test, lgbm_probs), 4)
lgbm_ks    = ks_statistic(y_test, lgbm_probs)

print(f"\n      LightGBM Results")
print(f"      Training time : {lgbm_time}s")
print(f"      AUC-ROC       : {lgbm_auc}")
print(f"      KS Statistic  : {lgbm_ks}")
print(f"\n      Classification Report:")
print(classification_report(y_test, lgbm_model.predict(X_test),
                             target_names=["Paid Off", "Defaulted"]))

# ─────────────────────────────────────────────────────────────────────────────
# FEATURE IMPORTANCE (LightGBM)
# ─────────────────────────────────────────────────────────────────────────────

print("\n      Top 15 Most Important Features (LightGBM):")
importance_df = pd.DataFrame({
    "feature":    X_train.columns,
    "importance": lgbm_model.feature_importances_,
}).sort_values("importance", ascending=False).head(15)

for _, row in importance_df.iterrows():
    bar = "#" * int(row["importance"] / importance_df["importance"].max() * 30)
    print(f"      {row['feature']:<35} {bar}")

# ─────────────────────────────────────────────────────────────────────────────
# COMPARE AND SAVE BEST MODEL
# ─────────────────────────────────────────────────────────────────────────────

print("\n[5/5] Comparing models and saving best ...")

print(f"\n      {'Model':<25} {'AUC-ROC':<12} {'KS Stat':<12}")
print(f"      {'-'*49}")
print(f"      {'Logistic Regression':<25} {lr_auc:<12} {lr_ks:<12}")
print(f"      {'LightGBM':<25} {lgbm_auc:<12} {lgbm_ks:<12}")

if lgbm_auc >= lr_auc:
    best_model      = lgbm_model
    best_model_name = "LightGBM"
    best_auc        = lgbm_auc
    best_ks         = lgbm_ks
else:
    best_model      = lr_model
    best_model_name = "Logistic Regression"
    best_auc        = lr_auc
    best_ks         = lr_ks

print(f"\n      Best model : {best_model_name}")

joblib.dump(best_model, "model.pkl")
size_mb = os.path.getsize("model.pkl") / 1024 ** 2
print(f"      Saved -> model.pkl  ({size_mb:.1f} MB)")

print("\n" + "=" * 60)
print(f"  Training complete.")
print(f"  Best model  : {best_model_name}")
print(f"  AUC-ROC     : {best_auc}  (target > 0.70)")
print(f"  KS Statistic: {best_ks}  (target > 0.40)")
print(f"  Saved as    : model.pkl")
print("=" * 60)

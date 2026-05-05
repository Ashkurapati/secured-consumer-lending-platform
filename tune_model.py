"""
tune_model.py
-------------
Step 6 of the Stepping Stones ML pipeline.

What this script does:
  1. Loads train.csv and test.csv
  2. Samples 200K rows from train for tuning (faster, still representative)
  3. Runs RandomizedSearchCV on LightGBM (20 combinations, 3-fold CV)
  4. Retrains best parameters on the FULL train set
  5. Evaluates on held-out test set
  6. Compares tuned model vs baseline LightGBM (AUC 0.7374, KS 0.3474)
  7. Saves the best model as model.pkl

Run on Darwin HPC:
    python tune_model.py

Expected time: 20-40 minutes on Darwin CPU nodes

Dependencies:
    pip install lightgbm==4.3.0 scikit-learn==1.3.0 pandas==2.0.3 numpy==1.24.0 joblib==1.3.0
"""

import pandas as pd
import numpy as np
import joblib
import os
import time

from scipy.stats import randint, uniform

from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, train_test_split
from sklearn.metrics import roc_auc_score, classification_report, roc_curve
from lightgbm import LGBMClassifier

TARGET    = "default"
SEED      = 42
TUNE_ROWS = 200000      # use 200K rows for tuning — fast and representative

# Baseline scores from train_model.py — used for final comparison
BASELINE_AUC = 0.7374
BASELINE_KS  = 0.3474

# ─────────────────────────────────────────────────────────────────────────────
# HELPER: KS STATISTIC
# ─────────────────────────────────────────────────────────────────────────────

def ks_statistic(y_true, y_prob):
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    return round(float(np.max(tpr - fpr)), 4)

# ─────────────────────────────────────────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────────────────────────────────────────

print("=" * 60)
print("  Stepping Stones -- Hyperparameter Tuning")
print("=" * 60)

print("\n[1/6] Loading train.csv and test.csv ...")

train_df = pd.read_csv("train.csv", low_memory=False)
test_df  = pd.read_csv("test.csv",  low_memory=False)

print(f"      Train : {len(train_df):,} rows x {len(train_df.columns)} columns")
print(f"      Test  : {len(test_df):,} rows x {len(test_df.columns)} columns")

# ─────────────────────────────────────────────────────────────────────────────
# SEPARATE FEATURES AND TARGET
# ─────────────────────────────────────────────────────────────────────────────

print("\n[2/6] Separating features and target ...")

X_train = train_df.drop(columns=[TARGET])
y_train = train_df[TARGET]

X_test  = test_df.drop(columns=[TARGET])
y_test  = test_df[TARGET]

print(f"      Features          : {X_train.shape[1]} columns")
print(f"      Train default rate: {y_train.mean():.2%}")
print(f"      Test  default rate: {y_test.mean():.2%}")

# ─────────────────────────────────────────────────────────────────────────────
# SAMPLE 200K ROWS FOR TUNING
# ─────────────────────────────────────────────────────────────────────────────

print(f"\n[3/6] Sampling {TUNE_ROWS:,} rows for hyperparameter search ...")

X_tune, _, y_tune, _ = train_test_split(
    X_train, y_train,
    train_size=TUNE_ROWS,
    random_state=SEED,
    stratify=y_train        # preserve 78/22 class ratio in the sample
)

print(f"      Tune sample : {len(X_tune):,} rows")
print(f"      Default rate in sample : {y_tune.mean():.2%}")
print(f"      (Best params found here will be retrained on full {len(X_train):,} rows)")

# ─────────────────────────────────────────────────────────────────────────────
# DEFINE SEARCH SPACE
# ─────────────────────────────────────────────────────────────────────────────

print("\n[4/6] Setting up Random Search ...")

param_dist = {
    "n_estimators":       randint(500, 2000),      # number of trees
    "learning_rate":      uniform(0.01, 0.09),     # 0.01 to 0.10
    "num_leaves":         randint(31, 128),        # tree complexity
    "min_child_samples":  randint(50, 200),        # overfitting control
    "subsample":          uniform(0.6, 0.4),       # 0.6 to 1.0 — rows per tree
    "colsample_bytree":   uniform(0.6, 0.4),       # 0.6 to 1.0 — features per tree
    "reg_alpha":          uniform(0.0, 0.5),       # L1 regularisation
    "reg_lambda":         uniform(0.0, 0.5),       # L2 regularisation
}

print(f"      Parameters to tune : {len(param_dist)}")
print(f"      Combinations tried : 20")
print(f"      Cross-validation   : 3-fold stratified")
print(f"      Tuning on          : {TUNE_ROWS:,} rows (not full dataset)")
print(f"      Scoring metric     : AUC-ROC")

# ─────────────────────────────────────────────────────────────────────────────
# RANDOM SEARCH ON 200K SAMPLE
# ─────────────────────────────────────────────────────────────────────────────

print("\n[5/6] Running Random Search ...")
print("      Expected time: 20-40 minutes\n")

base_model = LGBMClassifier(
    class_weight="balanced",
    random_state=SEED,
    n_jobs=1,               # 1 here — let RandomizedSearchCV parallelise instead
    verbose=-1,
    device="cpu",
)

cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED)

search = RandomizedSearchCV(
    estimator=base_model,
    param_distributions=param_dist,
    n_iter=20,               # reduced from 50 to 20 — fast, still effective
    scoring="roc_auc",
    cv=cv,
    random_state=SEED,
    n_jobs=-1,               # parallelise across Darwin CPU cores here
    verbose=2,
    refit=False,             # we will manually refit on full data below
)

t0 = time.time()
search.fit(X_tune, y_tune)
search_time = round(time.time() - t0, 1)

print(f"\n      Random Search complete in {search_time}s ({search_time/60:.1f} mins)")

# ─────────────────────────────────────────────────────────────────────────────
# BEST PARAMETERS
# ─────────────────────────────────────────────────────────────────────────────

print("\n      Best parameters found:")
for param, value in search.best_params_.items():
    print(f"        {param:<25} : {value}")

print(f"\n      Best CV AUC-ROC (on 200K sample) : {search.best_score_:.4f}")

# ─────────────────────────────────────────────────────────────────────────────
# RETRAIN BEST PARAMS ON FULL TRAINING SET
# ─────────────────────────────────────────────────────────────────────────────

print(f"\n      Retraining best parameters on full {len(X_train):,} rows ...")

t0 = time.time()

best_model = LGBMClassifier(
    **search.best_params_,
    class_weight="balanced",
    random_state=SEED,
    n_jobs=-1,               # full cores for final training
    verbose=-1,
    device="cpu",
)

best_model.fit(X_train, y_train)
retrain_time = round(time.time() - t0, 1)
print(f"      Retraining complete in {retrain_time}s")

# ─────────────────────────────────────────────────────────────────────────────
# EVALUATE ON TEST SET
# ─────────────────────────────────────────────────────────────────────────────

print("\n[6/6] Evaluating tuned model on test set ...")

tuned_probs = best_model.predict_proba(X_test)[:, 1]
tuned_auc   = round(roc_auc_score(y_test, tuned_probs), 4)
tuned_ks    = ks_statistic(y_test, tuned_probs)

print(f"\n      Classification Report:")
print(classification_report(y_test, best_model.predict(X_test),
                             target_names=["Paid Off", "Defaulted"]))

# ─────────────────────────────────────────────────────────────────────────────
# COMPARE TUNED VS BASELINE
# ─────────────────────────────────────────────────────────────────────────────

print(f"\n      {'Model':<25} {'AUC-ROC':<12} {'KS Stat':<12}")
print(f"      {'-'*49}")
print(f"      {'Baseline LightGBM':<25} {BASELINE_AUC:<12} {BASELINE_KS:<12}")
print(f"      {'Tuned LightGBM':<25} {tuned_auc:<12} {tuned_ks:<12}")

auc_improvement = round(tuned_auc - BASELINE_AUC, 4)
ks_improvement  = round(tuned_ks  - BASELINE_KS,  4)

print(f"\n      AUC improvement : {auc_improvement:+.4f}")
print(f"      KS  improvement : {ks_improvement:+.4f}")

# ─────────────────────────────────────────────────────────────────────────────
# SAVE BEST MODEL
# ─────────────────────────────────────────────────────────────────────────────

if tuned_auc >= BASELINE_AUC:
    joblib.dump(best_model, "model.pkl")
    size_mb = os.path.getsize("model.pkl") / 1024 ** 2
    print(f"\n      Tuned model is better — saved as model.pkl ({size_mb:.1f} MB)")
else:
    print(f"\n      Tuned model did not improve on baseline.")
    print(f"      Keeping existing model.pkl (baseline LightGBM AUC {BASELINE_AUC})")

print("\n" + "=" * 60)
print(f"  Tuning complete.")
print(f"  Tuned LightGBM  AUC-ROC : {tuned_auc}  (baseline: {BASELINE_AUC})")
print(f"  Tuned LightGBM  KS Stat : {tuned_ks}   (baseline: {BASELINE_KS})")
print("=" * 60)

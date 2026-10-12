"""
Train Final Model + Save Bundled Artifact (Phase B)
====================================================
Trains the final Ridge model on the S2 feature set (12 image features +
is_female), WITHOUT data leakage, and saves ONE bundled joblib artifact:

    outputs/models/model_pipeline.joblib

Bundle content:
    pipeline      -> trained sklearn Pipeline (StandardScaler + Ridge),
                     scaler already fitted INSIDE the pipeline
    feature_order -> explicit list of feature column names (contract lock)
    model_name    -> e.g. "Ridge (S2_image_gender)"
    thresholds    -> WHO thresholds per gender
    cv_metrics    -> out-of-fold 5-fold CV metrics (honest generalization est.)

NOTE: no SMOTE / no class_weight (regression task). sample_weight OFF
(default). Recall for the anemia class is reported from CV evaluation.
"""

import json
import os
import sys
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import joblib

from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold, cross_val_score, cross_val_predict
from sklearn.metrics import (mean_absolute_error, mean_squared_error, r2_score,
                             accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import BASE_PATH, RANDOM_STATE, MODELS_DIR
from data_loader import load_all_metadata
from image_processor import batch_process_patients

# ============================================================================
# FIXED CONFIG — matches the Fase A decision (S2, Ridge, drop age)
# ============================================================================
CV_FOLDS = 5
THRESHOLD_FEMALE = 12.0
THRESHOLD_MALE = 13.0
ARTIFACT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             "outputs", "models", "model_pipeline.joblib")

IMAGE_FEATURES = ["mean_R", "mean_G", "mean_B", "mean_L", "mean_a", "mean_b",
                  "mean_S", "mean_V", "std_a", "std_b", "std_S", "n_valid_px"]
FEATURE_ORDER = IMAGE_FEATURES + ["is_female"]  # S2_image_gender
FEATURE_SET_NAME = "S2_image_gender"


def build_dataset():
    """Load metadata, extract image features, build clean feature matrix."""
    df_meta = load_all_metadata()
    df = batch_process_patients(df_meta, BASE_PATH)

    df["gender_upper"] = df["gender"].astype(str).str.upper().str.strip()
    df["is_female"] = df["gender_upper"].apply(
        lambda x: 1 if x in ["F", "FEMALE"] else 0)

    assert "hgb" in df.columns
    y = df["hgb"].to_numpy(dtype=float)
    gender = df["gender_upper"].to_numpy()

    valid = ~np.isnan(y)
    df, y, gender = df[valid].reset_index(drop=True), y[valid], gender[valid]

    # Safety: no column in FEATURE_ORDER may be the target or a label artifact
    assert not ({"Hgb", "hgb"} & set(FEATURE_ORDER)), "Target bocor!"
    assert "is_anemic" not in FEATURE_ORDER and "anemia_risk_score" not in FEATURE_ORDER

    X = df[FEATURE_ORDER].to_numpy(dtype=float)
    assert X.shape[1] == len(FEATURE_ORDER) == 13
    return X, y, gender, df


def who_label(ypred_hgb, gender_arr):
    """WHO threshold per gender -> binary label."""
    labels = np.zeros(len(ypred_hgb), dtype=int)
    female = np.char.startswith(gender_arr.astype(str), "F")
    labels[female & (ypred_hgb < THRESHOLD_FEMALE)] = 1
    labels[~female & (ypred_hgb < THRESHOLD_MALE)] = 1
    return labels


def main():
    print("=" * 72)
    print("[FASE B] TRAIN FINAL MODEL & SAVE BUNDLED ARTIFACT")
    print("=" * 72)
    print(f"Feature set : {FEATURE_SET_NAME} ({len(FEATURE_ORDER)} fitur)")
    print(f"Feature order: {FEATURE_ORDER}")

    X, y, gender, df = build_dataset()
    n = len(y)
    print(f"\nDataset: {n} pasien | X shape: {X.shape}")

    # ------------------------------------------------------------------
    # Honest CV metrics on the FULL dataset (out-of-fold)
    # ------------------------------------------------------------------
    kf = KFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    pipe = Pipeline([("scaler", StandardScaler()), ("reg", Ridge(alpha=1.0))])

    cv_mae = -cross_val_score(pipe, X, y, cv=kf, scoring="neg_mean_absolute_error")
    cv_rmse = -cross_val_score(pipe, X, y, cv=kf, scoring="neg_root_mean_squared_error")
    cv_r2 = cross_val_score(pipe, X, y, cv=kf, scoring="r2")

    y_oof = cross_val_predict(pipe, X, y, cv=kf)
    y_true_label = who_label(y, gender)
    y_pred_label = who_label(y_oof, gender)

    cv_metrics = {
        "MAE_mean": round(float(np.mean(cv_mae)), 4),
        "MAE_std": round(float(np.std(cv_mae)), 4),
        "MAE_per_fold": [round(float(v), 4) for v in cv_mae],
        "RMSE_mean": round(float(np.mean(cv_rmse)), 4),
        "R2_mean": round(float(np.mean(cv_r2)), 4),
        "accuracy": round(float(accuracy_score(y_true_label, y_pred_label)), 4),
        "precision": round(float(precision_score(y_true_label, y_pred_label, zero_division=0)), 4),
        "recall": round(float(recall_score(y_true_label, y_pred_label)), 4),
        "f1": round(float(f1_score(y_true_label, y_pred_label, zero_division=0)), 4),
        "confusion_matrix": confusion_matrix(y_true_label, y_pred_label).tolist(),
    }
    print("\n[CV HONEST — out-of-fold, full dataset]")
    print(f"   MAE = {cv_metrics['MAE_mean']} ± {cv_metrics['MAE_std']} g/dL "
          f"(per-fold: {cv_metrics['MAE_per_fold']})")
    print(f"   RMSE = {cv_metrics['RMSE_mean']} | R² = {cv_metrics['R2_mean']}")
    print(f"   CLS: acc={cv_metrics['accuracy']} precision={cv_metrics['precision']} "
          f"recall={cv_metrics['recall']} f1={cv_metrics['f1']}")
    print(f"   Confusion matrix (rows=true, cols=pred): {cv_metrics['confusion_matrix']}")

    # ------------------------------------------------------------------
    # Train FINAL pipeline on ALL data (deployment artifact)
    # ------------------------------------------------------------------
    print("\n[FIT FINAL] StandardScaler + Ridge pada seluruh dataset...")
    pipe_final = Pipeline([("scaler", StandardScaler()), ("reg", Ridge(alpha=1.0))])
    pipe_final.fit(X, y)
    print("   ✓ fitted")

    y_full_pred = pipe_final.predict(X)
    train_mae = mean_absolute_error(y, y_full_pred)
    print(f"   (referensi) MAE training on full data: {train_mae:.4f} g/dL")

    # ------------------------------------------------------------------
    # Save ONE bundled artifact
    # ------------------------------------------------------------------
    artifact = {
        "pipeline": pipe_final,                 # scaler fitted INSIDE
        "feature_order": list(FEATURE_ORDER),   # contract lock
        "feature_set": FEATURE_SET_NAME,
        "model_name": "Ridge",
        "thresholds": {"female": THRESHOLD_FEMALE, "male": THRESHOLD_MALE},
        "cv_metrics": cv_metrics,
        "n_samples_trained": int(n),
    }
    os.makedirs(os.path.dirname(ARTIFACT_PATH), exist_ok=True)
    joblib.dump(artifact, ARTIFACT_PATH)
    size_kb = os.path.getsize(ARTIFACT_PATH) / 1024
    print(f"\n💾 Artifact tersimpan: {ARTIFACT_PATH} ({size_kb:.2f} KB)")
    print(f"   Bundle keys: {list(artifact.keys())}")

    # ------------------------------------------------------------------
    # Load-Back sanity: predict must be identical after re-load
    # ------------------------------------------------------------------
    re = joblib.load(ARTIFACT_PATH)
    assert list(re["pipeline"].named_steps.keys()) == ["scaler", "reg"]
    assert re["feature_order"] == FEATURE_ORDER

    X_samp = X[:5]
    p_first = pipe_final.predict(X_samp)
    p_reload = re["pipeline"].predict(X_samp)
    assert np.allclose(p_first, p_reload), "Load-back predict TIDAK identik!"
    print("\n[LOAD-BACK PASS] predict sebelum vs setelah reload:")
    print("   first  :", np.round(p_first, 4))
    print("   reload :", np.round(p_reload, 4))
    print("   np.allclose ->", np.allclose(p_first, p_reload), "(PASS)")

    print("\n✅ FASE B SELESAI.")


if __name__ == "__main__":
    main()
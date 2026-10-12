"""
Honest Model Evaluation (Phase A)
=================================
Evaluates candidate models WITHOUT data leakage.

Key design rules (implemented deliberately):
  1. Target column ('hgb' / 'Hgb') is NEVER part of the feature matrix.
     A defensive assertion guards this before any training.
  2. Label ('is_anemic') and derived-leaky features ('anemia_risk_score')
     are excluded from the feature set. Only pure image + demographic
     features are used.
  3. No manual age normalization over the full dataset. 'age' is included
     RAW; scaling is handled ONLY by the StandardScaler INSIDE each
     sklearn Pipeline, which is fit on training folds only.
  4. Per-fold pipeline: 5-fold CV. Every fold's scaler learns from its
     own training fold (no test leakage by construction).
  5. Classification metrics are computed after the WHO gender-specific
     threshold is applied to out-of-fold predicted Hgb.
     This reports Sensitivity/Recall honestly, without SMOTE tricks.
"""

import json
import os
import sys
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from sklearn.linear_model import Ridge
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold, cross_val_predict, cross_val_score
from sklearn.metrics import (mean_absolute_error, mean_squared_error, r2_score,
                             accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix, classification_report)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import BASE_PATH, RANDOM_STATE
from data_loader import load_all_metadata
from image_processor import batch_process_patients

# ============================================================================
# CONFIG
# ============================================================================
CV_FOLDS = 5
THRESHOLD_FEMALE = 12.0
THRESHOLD_MALE = 13.0
OUTPUT_REPORT = os.path.join("outputs", "evaluation_report.json")

# ============================================================================
# FEATURE SETS (explicit, reuse at inference)
# ============================================================================
IMAGE_FEATURES = ["mean_R", "mean_G", "mean_B", "mean_L", "mean_a", "mean_b",
                  "mean_S", "mean_V", "std_a", "std_b", "std_S", "n_valid_px"]

FEATURE_SETS = {
    "S1_image_only": IMAGE_FEATURES,
    "S2_image_gender": IMAGE_FEATURES + ["is_female"],
    "S3_image_gender_age": IMAGE_FEATURES + ["is_female", "age"],
}

MODELS = {
    "Ridge": Pipeline([("scaler", StandardScaler()),
                       ("reg", Ridge(alpha=1.0))]),
    "GradientBoosting": Pipeline([("scaler", StandardScaler()),
                                  ("reg", GradientBoostingRegressor(
                                      n_estimators=300, learning_rate=0.05,
                                      max_depth=3, random_state=RANDOM_STATE))]),
    "RandomForest": Pipeline([("scaler", StandardScaler()),
                              ("reg", RandomForestRegressor(
                                  n_estimators=500, max_depth=12,
                                  n_jobs=-1, random_state=RANDOM_STATE))]),
}


def _verify_no_leakage(feature_cols, df):
    """Defensive guard: target and label-derived features must NOT leak here."""
    present = list(df.columns)
    print(f"\n[VERIFY] Actual columns ({len(present)}): {present}\n")

    targets = {"Hgb", "hgb"}
    leak_target = targets & set(feature_cols)
    assert not leak_target, f"❌ FAULT: target kolom bocor jadi fitur: {leak_target}"

    leak_label = {"is_anemic", "anemia_risk_score"} & set(feature_cols)
    assert not leak_label, f"❌ FAULT: fitur turunan bocor: {leak_label}"

    print("✅ ASSERTION LULUS: tidak ada leakage target/label pada fitur.\n")


def build_dataset():
    """Load metadata, extract image features, build clean feature columns."""
    df_meta = load_all_metadata()
    df_feat = batch_process_patients(df_meta, BASE_PATH)

    df = df_feat.copy()

    # Clean gender encoding
    df["gender_upper"] = df["gender"].astype(str).str.upper().str.strip()
    df["is_female"] = df["gender_upper"].apply(
        lambda x: 1 if x in ["F", "FEMALE"] else 0)

    # Keep raw age (no whole-dataset normalization here; scaler inside pipeline)
    df["age"] = df["age"].astype(float)

    # Target: actual Hemoglobin (lowercase 'hgb' produced by extraction)
    assert "hgb" in df.columns, "Target 'hgb' tidak ditemukan!"
    y = df["hgb"].to_numpy(dtype=float)
    gender = df["gender_upper"].to_numpy()

    # Drop rows with NaN target (shouldn't exist after load, but be safe)
    valid = ~np.isnan(y)
    df, y, gender = df[valid].reset_index(drop=True), y[valid], gender[valid]

    print(f"\n[BUILD] Dataset final: {len(df)} pasien | max leakage-safe rows: {valid.sum()}")
    return df, y, gender


def who_label(ypred_hgb, gender_arr):
    """Map predicted Hgb to binary anemia label using WHO gender thresholds."""
    labels = np.zeros(len(ypred_hgb), dtype=int)
    female = np.char.startswith(gender_arr.astype(str), "F")
    labels[female & (ypred_hgb < THRESHOLD_FEMALE)] = 1
    labels[~female & (ypred_hgb < THRESHOLD_MALE)] = 1
    return labels


def run_cv_metrics(X, y, gender, model, model_name):
    """5-fold CV: regression metrics from OOF preds + classification metrics."""
    kf = KFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    cv_mae = cross_val_score(model, X, y, cv=kf,
                             scoring="neg_mean_absolute_error")
    cv_rmse = cross_val_score(model, X, y, cv=kf,
                              scoring="neg_root_mean_squared_error")
    cv_r2 = cross_val_score(model, X, y, cv=kf, scoring="r2")

    y_pred_oof = cross_val_predict(model, X, y, cv=kf)

    reg = {
        "MAE": round(abs(float(np.mean(cv_mae))), 4),
        "MAE_std": round(float(np.std(cv_mae)), 4),
        "RMSE": round(float(abs(np.mean(cv_rmse))), 4),
        "R2": round(float(np.mean(cv_r2)), 4),
    }

    # Classification after WHO threshold
    y_label = who_label(y, gender)
    pred_label = who_label(y_pred_oof, gender)

    cls = {
        "accuracy": round(float(accuracy_score(y_label, pred_label)), 4),
        "precision": round(float(precision_score(y_label, pred_label, zero_division=0)), 4),
        "recall": round(float(recall_score(y_label, pred_label)), 4),
        "f1": round(float(f1_score(y_label, pred_label, zero_division=0)), 4),
        "confusion_matrix": confusion_matrix(y_label, pred_label).tolist(),
    }

    print(f"   ─ {model_name:<16} MAE={reg['MAE']:>6}±{reg['MAE_std']:<6} "
          f"RMSE={reg['RMSE']:>6} R²={reg['R2']:>6} | "
          f"Acc={cls['accuracy']:.3f} Recall={cls['recall']:.3f}")
    return reg, cls


def main():
    print("=" * 72)
    print("[FASE A] HONEST EVALUATION — tanpa data leakage")
    print("=" * 72)

    df, y, gender = build_dataset()

    report = {
        "dataset_size": int(len(df)),
        "thresholds": {"female": THRESHOLD_FEMALE, "male": THRESHOLD_MALE},
        "cv_folds": CV_FOLDS,
        "feature_sets": {},
    }

    for set_name, feature_cols in FEATURE_SETS.items():
        print("\n" + "=" * 72)
        print(f"[SET] {set_name} — {len(feature_cols)} fitur")
        print("=" * 72)

        _verify_no_leakage(feature_cols, df)

        X = df[feature_cols].to_numpy(dtype=float)
        assert X.shape[1] == len(feature_cols)

        report["feature_sets"][set_name] = {
            "feature_order": list(feature_cols),
            "models": {},
        }
        # Save per-set feature order for future use
        report["feature_sets"][set_name]["X_shape"] = list(X.shape)

        for model_name, model in MODELS.items():
            reg, cls = run_cv_metrics(X, y, gender, model, model_name)
            report["feature_sets"][set_name]["models"][model_name] = {
                "regression": reg, "classification": cls}

    # ------------------------------------------------------------------
    # SUMMARY TABLE (MAE per feature set, best across the 3 models)
    # ------------------------------------------------------------------
    print("\n" + "=" * 72)
    print("[SUMMARY] MAE (mean ± std) tiap feature-set — terbaik tiap model")
    print("=" * 72)

    summary_rows = []
    for set_name in FEATURE_SETS:
        for model_name in MODELS:
            m = report["feature_sets"][set_name]["models"][model_name]
            summary_rows.append((set_name, model_name, m["regression"]["MAE"],
                                 m["regression"]["R2"],
                                 m["classification"]["recall"]))
            print(f"   {set_name:<22} {model_name:<16} "
                  f"MAE={m['regression']['MAE']:<7} "
                  f"R²={m['regression']['R2']:<7} "
                  f"Recall={m['classification']['recall']:.3f}")

    # Best MAE per feature set
    print("\n[BEST MAE per feature-set]")
    best_per_set = {}
    for set_name in FEATURE_SETS:
        best = min(report["feature_sets"][set_name]["models"].items(),
                   key=lambda kv: kv[1]["regression"]["MAE"])
        best_per_set[set_name] = (best[0], best[1]["regression"]["MAE"])
        print(f"   {set_name:<22} → {best[0]:<16} MAE={best[1]['regression']['MAE']}")

    report["best_per_set"] = {k: {"model": v[0], "MAE": v[1]} for k, v in best_per_set.items()}

    # ------------------------------------------------------------------
    # A/B DECISION PREVIEW (age effect: S3 vs S2)
    # ------------------------------------------------------------------
    s2_best = best_per_set["S2_image_gender"][1]
    s3_best = best_per_set["S3_image_gender_age"][1]
    delta = s3_best - s2_best
    report["age_delta_MAE_S3_vs_S2"] = round(delta, 4)
    print(f"\n[DECISION PREVIEW] ΔMAE (S3−S2) = {delta:+.4f} "
          f"(<0 berarti 'age' menurunkan MAE)")

    # Save report
    os.makedirs(os.path.dirname(OUTPUT_REPORT), exist_ok=True)
    with open(OUTPUT_REPORT, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n📄 Report tersimpan: {OUTPUT_REPORT}")


if __name__ == "__main__":
    main()
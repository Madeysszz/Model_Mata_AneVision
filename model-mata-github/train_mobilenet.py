"""
Train MobileNetV2 Feature Extractor + Classifier (Phase E2)
============================================================
Uses pretrained ImageNet MobileNetV2 as FIXED feature extractor (all base
layers frozen). Flattens the 7x7x1280 output to 62720 features, then trains
a Logistic Regression classifier. This approach:
  - Runs on CPU in ~5-10 minutes (no fine-tuning of base layers)
  - Still leverages pretrained ImageNet features (1280 rich features >> 13 manual)
  - Produces joblib artifacts ready for inference

Strategy for ensemble with Ridge:
  - Ridge: predict Hgb (regression) → threshold → status
  - MobileNetV2: predict anemia probability → threshold → status
  - Ensemble: anemia if either model predicts anemia
"""

import os
import sys
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import cv2
import joblib

from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, recall_score, precision_score,
                             f1_score, confusion_matrix, classification_report)

import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array, load_img

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import BASE_PATH

# ============================================================================
# CONFIG
# ============================================================================
IMG_SIZE = 224
BATCH_SIZE = 32
N_SPLITS = 5
RANDOM_STATE = 42
OUTPUT_DIR = os.path.join("outputs", "models")
ARTIFACT_PATH = os.path.join(OUTPUT_DIR, "mobilenet_anemia.joblib")
EXTRACTOR_PATH = os.path.join(OUTPUT_DIR, "mobilenet_extractor.joblib")


def load_dataset():
    """Load all valid image paths + labels (is_anemic)."""
    records = []
    
    for country in ["India", "Italy"]:
        excel_path = os.path.join(BASE_PATH, country, f"{country}.xlsx")
        if not os.path.exists(excel_path):
            continue
            
        df = pd.read_excel(excel_path, usecols=["Number", "Hgb", "Gender", "Age"])
        df["country"] = country
        
        for _, row in df.iterrows():
            num = int(row["Number"])
            hgb = row["Hgb"]
            gender = str(row["Gender"]).strip().upper()
            
            # Skip missing Hgb
            if pd.isna(hgb):
                continue
            # Handle comma decimals (Italian data uses '15,1' style)
            if isinstance(hgb, str):
                hgb = hgb.replace(",", ".").strip()
            try:
                hgb = float(hgb)
            except (ValueError, TypeError):
                # Some cells contain junk (e.g. '_'); skip that row
                continue
            
            # Find forniceal image
            folder = os.path.join(BASE_PATH, country, str(num))
            if not os.path.isdir(folder):
                continue
            
            img_path = None
            for f in os.listdir(folder):
                if "forniceal" in f.lower() and f.lower().endswith(('.png', '.jpg', '.jpeg')):
                    img_path = os.path.join(folder, f)
                    break
            
            if img_path is None:
                continue
            
            # Validate image actually loads (some files are corrupt)
            probe = cv2.imread(img_path)
            if probe is None:
                continue
            
            # Label: WHO threshold
            is_female = gender.startswith(("F", "P"))
            threshold = 12.0 if is_female else 13.0
            is_anemic = 1 if hgb < threshold else 0
            
            records.append({
                "img_path": img_path,
                "hgb": hgb,
                "gender": "F" if is_female else "M",
                "is_female": int(is_female),
                "is_anemic": is_anemic,
                "country": country,
                "patient_id": num,
            })
    
    df = pd.DataFrame(records)
    print(f"Dataset: {len(df)} gambar valid")
    print(f"  Anemic: {(df.is_anemic==1).sum()} | Normal: {(df.is_anemic==0).sum()}")
    print(f"  India: {(df.country=='India').sum()} | Italy: {(df.country=='Italy').sum()}")
    return df


def build_extractor():
    """MobileNetV2 pretrained (frozen), NO pooling — naive flatten (62720-d).
    Empirical CV comparison showed flatten (62720-d) generalizes better than
    GlobalAveragePooling (1280-d) on this small dataset (211 samples):
      flatten: acc 0.734, recall 0.800 | GAP: acc 0.673, recall 0.589
    Spatial information from the 7×7 grid is retained and helps Ridge/LogReg.
    See: experiment documented in session (GAP rolled back)."""
    base = MobileNetV2(
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
        include_top=False,
        weights="imagenet"
    )
    base.trainable = False  # freeze 100% — no fine-tuning

    flat = tf.keras.layers.Flatten()(base.output)
    extractor = tf.keras.Model(inputs=base.input, outputs=flat)
    return extractor


def extract_features_batch(extractor, img_paths):
    """Extract 62720 features per image using frozen MobileNetV2.
    Uses cv2 directly (more robust than PIL for files with iCCP CRC errors)."""
    features = []
    
    for path in img_paths:
        img = cv2.imread(path)                    # BGR
        if img is None:
            raise ValueError(f"Gagal membaca: {path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        arr = np.expand_dims(img.astype(np.float32), axis=0)
        arr = preprocess_input(arr)
        
        feat = extractor.predict(arr, verbose=0)
        feat = feat.flatten()  # (1,7,7,1280) → (62720,)
        features.append(feat)
    
    return np.array(features)


def train_classifier(X, y):
    """Train LogisticRegression on MobileNetV2 features (fast, no GPU needed)."""
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    clf = LogisticRegression(
        C=1.0,
        max_iter=1000,
        random_state=RANDOM_STATE,
        class_weight="balanced"  # handle imbalance
    )
    clf.fit(X_scaled, y)
    
    return clf, scaler


def evaluate_cv(extractor, df):
    """5-fold Stratified CV using LogisticRegression on MobileNetV2 features."""
    print("\n" + "="*70)
    print("[PHASE E2] MobileNetV2 Feature Extraction + Classifier CV")
    print("="*70)
    
    # Extract ALL features first (one-time cost)
    print("\n[1/3] Extracting features from MobileNetV2 (frozen base)...")
    features = extract_features_batch(extractor, df["img_path"].values)
    y = df["is_anemic"].values
    print(f"  Features shape: {features.shape} (n_samples × 62720)")
    
    # Stratified K-Fold
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    
    all_metrics = []
    fold_models = []
    
    print(f"\n[2/3] Training classifier ({N_SPLITS}-fold Stratified CV)...")
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(features, y)):
        X_train, X_val = features[train_idx], features[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        clf, scaler = train_classifier(X_train, y_train)
        
        X_val_scaled = scaler.transform(X_val)
        y_pred = clf.predict(X_val_scaled)
        y_prob = clf.predict_proba(X_val_scaled)[:, 1]
        
        acc = accuracy_score(y_val, y_pred)
        rec = recall_score(y_val, y_pred)
        prec = precision_score(y_val, y_pred)
        f1 = f1_score(y_val, y_pred)
        cm = confusion_matrix(y_val, y_pred)
        
        all_metrics.append({
            "fold": fold + 1,
            "accuracy": acc,
            "recall": rec,
            "precision": prec,
            "f1": f1,
            "confusion_matrix": cm.tolist(),
        })
        fold_models.append((clf, scaler))
        
        print(f"  Fold {fold+1}: Acc={acc:.3f} Recall={rec:.3f} Prec={prec:.3f} F1={f1:.3f}")
        print(f"           CM: {cm.tolist()}")
    
    # Summary
    metrics_df = pd.DataFrame(all_metrics)
    print(f"\n[3/3] Summary (mean ± std across {N_SPLITS} folds):")
    for col in ["accuracy", "recall", "precision", "f1"]:
        vals = metrics_df[col]
        print(f"  {col:>10}: {vals.mean():.3f} ± {vals.std():.3f}")
    
    return features, y, fold_models, all_metrics


def train_final_model(extractor, features, y, fold_models):
    """Train final classifier on ALL data (for deployment artifact)."""
    print("\n[FINAL] Training classifier on full dataset...")
    
    # Use the best fold's scaler approach (fit on all data)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(features)
    
    clf = LogisticRegression(
        C=1.0,
        max_iter=1000,
        random_state=RANDOM_STATE,
        class_weight="balanced"
    )
    clf.fit(X_scaled, y)
    
    # Verify
    y_pred = clf.predict(X_scaled)
    train_acc = accuracy_score(y, y_pred)
    print(f"  Training accuracy (full data): {train_acc:.3f}")
    
    # Save artifact: dict with scaler + clf + feature_names
    artifact = {
        "scaler": scaler,
        "classifier": clf,
        "model_name": "MobileNetV2(frozen) + LogisticRegression",
        "feature_dim": features.shape[1],
        "img_size": IMG_SIZE,
        "threshold": 0.5,  # for classification
    }
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    joblib.dump(artifact, ARTIFACT_PATH)
    size_kb = os.path.getsize(ARTIFACT_PATH) / 1024
    print(f"  Saved: {ARTIFACT_PATH} ({size_kb:.0f} KB)")
    
    # Also save extractor separately for inference
    extractor_artifact = {
        "model": extractor,
        "img_size": IMG_SIZE,
    }
    joblib.dump(extractor_artifact, EXTRACTOR_PATH)
    size_kb2 = os.path.getsize(EXTRACTOR_PATH) / 1024
    print(f"  Saved: {EXTRACTOR_PATH} ({size_kb2:.0f} KB)")
    
    return clf, scaler


def main():
    print("="*70)
    print("[PHASE E2] MobileNetV2 Feature Extraction + Classifier Training")
    print("="*70)
    print(f"TensorFlow: {tf.__version__}")
    print(f"GPU: {tf.config.list_physical_devices('GPU') or 'CPU only'}")
    
    # Load dataset
    df = load_dataset()
    
    # Build frozen extractor
    print("\nLoading MobileNetV2 pretrained (ImageNet, frozen)...")
    extractor = build_extractor()
    print(f"  Base layers: {len(extractor.layers)} (all frozen)")
    print(f"  Output shape: {extractor.output_shape}")
    
    # CV evaluation
    features, y, fold_models, cv_metrics = evaluate_cv(extractor, df)
    
    # Train final model on all data
    clf, scaler = train_final_model(extractor, features, y, fold_models)
    
    # Save CV metrics
    metrics_path = os.path.join(OUTPUT_DIR, "mobilenet_cv_metrics.json")
    import json
    with open(metrics_path, "w") as f:
        json.dump(cv_metrics, f, indent=2)
    print(f"\n  CV metrics saved: {metrics_path}")
    
    print("\n" + "="*70)
    print("✅ PHASE E2 COMPLETE — MobileNetV2 classifier trained!")
    print("="*70)


if __name__ == "__main__":
    main()

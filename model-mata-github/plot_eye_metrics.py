from pathlib import Path

import cv2
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.base import clone
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
)
from sklearn.model_selection import StratifiedKFold
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

from feature_extraction import extract_features_advanced_masking
from train_mobilenet import load_dataset


ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "outputs" / "models"
OUTPUT = ROOT / "outputs" / "plots" / "eye_precision_recall_f1.png"


def main():
    ridge_bundle = joblib.load(MODEL_DIR / "model_pipeline.joblib")
    cnn_bundle = joblib.load(MODEL_DIR / "mobilenet_anemia.joblib")
    extractor = joblib.load(MODEL_DIR / "mobilenet_extractor.joblib")["model"]
    records = load_dataset()
    ridge_inputs = []
    cnn_inputs = []

    for row in records.itertuples():
        image = cv2.imread(row.img_path)
        features = extract_features_advanced_masking(image)
        if features is None:
            raise ValueError(f"Fitur citra tidak tersedia: {row.img_path}")
        ridge_inputs.append([features[name] if name != "is_female" else row.is_female
                             for name in ridge_bundle["feature_order"]])
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, (cnn_bundle["img_size"], cnn_bundle["img_size"]))
        cnn_inputs.append(preprocess_input(resized.astype(np.float32)))

    x_ridge = np.asarray(ridge_inputs, dtype=float)
    x_cnn = extractor.predict(np.asarray(cnn_inputs), batch_size=16, verbose=0)
    x_cnn = x_cnn.reshape(len(records), -1)
    y_hb = records["hgb"].to_numpy(dtype=float)
    y_class = records["is_anemic"].to_numpy(dtype=int)
    hb_threshold = np.where(records["is_female"].to_numpy(dtype=bool), 12.0, 13.0)
    ridge_hb = np.empty(len(records))
    cnn_probability = np.empty(len(records))

    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    for train, valid in folds.split(x_ridge, y_class):
        ridge = clone(ridge_bundle["pipeline"]).fit(x_ridge[train], y_hb[train])
        scaler = clone(cnn_bundle["scaler"]).fit(x_cnn[train])
        classifier = clone(cnn_bundle["classifier"]).fit(
            scaler.transform(x_cnn[train]), y_class[train]
        )
        ridge_hb[valid] = ridge.predict(x_ridge[valid])
        cnn_probability[valid] = classifier.predict_proba(
            scaler.transform(x_cnn[valid])
        )[:, 1]

    ridge_score = hb_threshold - ridge_hb
    ridge_status = ridge_hb < hb_threshold
    cutoffs = np.linspace(0, 1, 201)
    cnn_precision = []
    cnn_recall = []
    cnn_f1 = []
    combined_precision = []
    combined_recall = []
    combined_f1 = []

    for cutoff in cutoffs:
        cnn_status = cnn_probability >= cutoff
        combined_status = ridge_status | cnn_status
        cnn_precision.append(precision_score(y_class, cnn_status, zero_division=0))
        cnn_recall.append(recall_score(y_class, cnn_status, zero_division=0))
        cnn_f1.append(f1_score(y_class, cnn_status, zero_division=0))
        combined_precision.append(precision_score(y_class, combined_status, zero_division=0))
        combined_recall.append(recall_score(y_class, combined_status, zero_division=0))
        combined_f1.append(f1_score(y_class, combined_status, zero_division=0))

    ridge_precision, ridge_recall, _ = precision_recall_curve(y_class, ridge_score)
    mobile_precision, mobile_recall, _ = precision_recall_curve(y_class, cnn_probability)
    ridge_f1 = f1_score(y_class, ridge_status)

    fig, (ax_pr, ax_f1) = plt.subplots(1, 2, figsize=(13, 5))
    ax_pr.step(ridge_recall, ridge_precision, where="post",
               label=f"Ridge (AP={average_precision_score(y_class, ridge_score):.3f})")
    ax_pr.step(mobile_recall, mobile_precision, where="post",
               label=f"MobileNetV2 (AP={average_precision_score(y_class, cnn_probability):.3f})")
    ax_pr.plot(combined_recall, combined_precision,
               label="Gabungan (ambang CNN berubah)")
    ax_pr.set(xlabel="Sensitivitas (recall anemia)", ylabel="Presisi anemia",
              title="Kurva presisi–recall", xlim=(0, 1.01), ylim=(0, 1.01))
    ax_pr.legend(loc="lower left")
    ax_pr.grid(alpha=0.25)

    ax_f1.plot(cutoffs, cnn_f1, label="MobileNetV2")
    ax_f1.plot(cutoffs, combined_f1, label="Gabungan")
    ax_f1.axhline(ridge_f1, color="gray", linestyle="--", label="Ridge (ambang Hb tetap)")
    ax_f1.axvline(0.5, color="black", linestyle=":", label="Ambang dipakai: 0,5")
    ax_f1.set(xlabel="Ambang probabilitas MobileNetV2", ylabel="F1 anemia",
              title="F1 terhadap ambang probabilitas", xlim=(0, 1), ylim=(0, 1))
    ax_f1.legend(loc="lower left")
    ax_f1.grid(alpha=0.25)

    fig.suptitle(f"Evaluasi out-of-fold 5-fold pada {len(records)} pasien")
    fig.tight_layout()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT, dpi=180, bbox_inches="tight")
    plt.close(fig)
    index = np.where(np.isclose(cutoffs, 0.5))[0][0]
    print(f"Tersimpan: {OUTPUT}")
    print(f"Pada ambang 0,5: Ridge F1={ridge_f1:.4f}; "
          f"MobileNetV2 F1={cnn_f1[index]:.4f}; Gabungan F1={combined_f1[index]:.4f}")


if __name__ == "__main__":
    main()

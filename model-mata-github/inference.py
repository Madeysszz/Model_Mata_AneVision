"""
Real-Time Inference Module (Ensemble: Ridge + MobileNetV2)
==========================================================
Turns a conjunctival eye image into a structured prediction dictionary
ready to be JSON-serialized.

Two models:
  1. Ridge (tabular, on 13 manual color features)  -> predicts Hgb (regression)
  2. MobileNetV2 (frozen ImageNet + LogisticRegression) -> predicts anemia prob.
Ensemble marks anemia if either model predicts anemia.

Backend usage (no web framework needed):
    from inference import predict_from_image
    result = predict_from_image(img_bgr, gender="F")

Dependencies: numpy, opencv-python, scikit-learn, joblib, tensorflow/keras
TensorFlow is imported lazily (only when the CNN path is used), so a
Ridge-only call stays lightweight.

Contract fields:
  hgb_predicted, status, status_ridge, status_cnn, confidence,
  threshold_used, margin_ke_threshold, gender, source, disclaimer
"""

import os
import joblib
import numpy as np
import cv2

from feature_extraction import extract_features_advanced_masking

# ---------------------------------------------------------------------------
# Load Ridge artifact (core, always required)
# ---------------------------------------------------------------------------
_ARTIFACT_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "outputs", "models", "model_pipeline.joblib")

_ARTIFACT = joblib.load(_ARTIFACT_PATH)
MODEL = _ARTIFACT["pipeline"]
FEATURE_ORDER = list(_ARTIFACT["feature_order"])
THRESHOLDS = _ARTIFACT["thresholds"]  # {"female": 12.0, "male": 13.0}

DISCLAIMER = ("Hasil ini untuk skrining edukatif, bukan pengganti "
              "diagnosis medis")

# CNN artifact paths (lazy-loaded)
_CNN_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "outputs", "models")
_CNN_CLF_PATH = os.path.join(_CNN_DIR, "mobilenet_anemia.joblib")
_CNN_EXTRACTOR_PATH = os.path.join(_CNN_DIR, "mobilenet_extractor.joblib")

# CNN lazy-load cache
_cnn_cache = None


def _load_cnn():
    """Lazy-load CNN extractor + classifier. Returns None if unavailable or fails."""
    global _cnn_cache
    if _cnn_cache is not None:
        return _cnn_cache

    try:
        import tensorflow as tf  # heavy; imported lazily
        from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

        art = joblib.load(_CNN_CLF_PATH)
        ext = joblib.load(_CNN_EXTRACTOR_PATH)
        _cnn_cache = {
            "preprocess_input": preprocess_input,
            "extractor": ext["model"],
            "scaler": art["scaler"],
            "classifier": art["classifier"],
            "img_size": int(art["img_size"]),
            "threshold": float(art["threshold"]),
            "cnn_ok": True,
        }
        return _cnn_cache
    except Exception:
        _cnn_cache = {"cnn_ok": False}
        return _cnn_cache


def _gender_info(gender):
    gender_str = str(gender).strip().upper()
    is_female = gender_str.startswith(("F", "P"))
    return is_female, ("Wanita" if is_female else "Pria")


def _predict_ridge(feats, is_female):
    """Ridge path: feature vector -> Hgb -> status."""
    try:
        x = np.array([[float(feats[k]) for k in FEATURE_ORDER
                       if k != "is_female"] + [1 if is_female else 0]],
                     dtype=np.float64)
    except KeyError as exc:
        return {"error": f"Fitur tidak lengkap dari ekstraksi: {exc}"}
    assert x.shape[1] == len(FEATURE_ORDER), "Feature-order mismatch!"

    hgb_pred = float(MODEL.predict(x)[0])
    threshold = THRESHOLDS["female"] if is_female else THRESHOLDS["male"]
    status = "Anemia" if hgb_pred < threshold else "Normal"
    return {
        "hgb_predicted": round(hgb_pred, 2),
        "status": status,
        "threshold_used": threshold,
        "margin_ke_threshold": round(hgb_pred - threshold, 2),
    }


def _predict_cnn(img_bgr):
    """CNN path: raw image -> MobileNetV2 features -> anemia probability."""
    cnn = _load_cnn()
    if not cnn.get("cnn_ok", False):
        return None

    img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (cnn["img_size"], cnn["img_size"]))
    arr = np.expand_dims(img.astype(np.float32), axis=0)
    arr = cnn["preprocess_input"](arr)

    feat = cnn["extractor"].predict(arr, verbose=0).flatten().reshape(1, -1)
    feat_scaled = cnn["scaler"].transform(feat)
    prob = float(cnn["classifier"].predict_proba(feat_scaled)[0][1])
    status = "Anemia" if prob >= cnn["threshold"] else "Normal"
    return {"probability": prob, "status": status}


def predict_from_image(img_bgr, gender):
    """
    Predict Hemoglobin + anemia status from a conjunctival image.

    Args:
        img_bgr (numpy.ndarray): BGR image (cv2.imread/decoded upload).
        gender (str): "F"/"Female"/"Perempuan" -> wanita; else pria.

    Returns:
        dict (JSON-serializable), or {"error": ...} on invalid image.
    """
    # 1) Guard invalid input
    if img_bgr is None or (isinstance(img_bgr, np.ndarray) and img_bgr.size == 0):
        return {"error": "Gambar tidak valid / ROI konjungtiva tidak terdeteksi"}

    # 2) Masking features for Ridge
    feats = extract_features_advanced_masking(img_bgr)
    if feats is None:
        return {"error": "Gambar tidak valid / ROI konjungtiva tidak terdeteksi"}

    is_female, gender_label = _gender_info(gender)

    # 3) Ridge path (always available)
    ridge = _predict_ridge(feats, is_female)
    if "error" in ridge:
        return ridge

    # 4) CNN path (best-effort; fall back to Ridge if unavailable)
    cnn = _predict_cnn(img_bgr)
    cnn_ok = cnn is not None

    # A disagreement is flagged as anemia to prioritize sensitivity.
    if cnn_ok:
        votes = [ridge["status"], cnn["status"]]
        agree = (ridge["status"] == cnn["status"])
        models_disagree = not agree

        if agree:
            final_status = ridge["status"]
        else:
            # Tie: conservative toward screening (flag for confirmation)
            final_status = "Anemia"
            models_disagree = True

        # Confidence: blend Ridge margin + CNN confidence
        margin = abs(float(ridge["margin_ke_threshold"]))
        conf_ridge = min(margin / 5.0, 1.0)                     # 5 g/dL full conf
        conf_cnn = max(cnn["probability"], 1 - cnn["probability"])  # 0..1
        confidence = round((conf_ridge + conf_cnn) / 2, 3)
        source = "ensemble_Ridge+MobileNetV2"
    else:
        final_status = ridge["status"]
        models_disagree = False
        confidence = round(min(abs(float(ridge["margin_ke_threshold"])) / 5.0, 1.0), 3)
        source = "Ridge_only"

    output = {
        "hgb_predicted": ridge["hgb_predicted"],
        "status": final_status,
        "status_ridge": ridge["status"],
        "threshold_used": ridge["threshold_used"],
        "margin_ke_threshold": ridge["margin_ke_threshold"],
        "gender": gender_label,
        "models_disagree": models_disagree,
        "source": source,
        "confidence": confidence,
        "disclaimer": DISCLAIMER,
    }
    if cnn_ok:
        output["status_cnn"] = cnn["status"]
        output["cnn_probability"] = round(cnn["probability"], 3)

    return output



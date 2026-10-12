from pathlib import Path

import cv2
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

from feature_extraction import extract_features_advanced_masking


ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "outputs" / "models"
INPUT_DIR = ROOT / "test_images"
OUTPUT_DIR = ROOT / "test_results_eye_attention"


def main():
    classifier_bundle = joblib.load(MODEL_DIR / "mobilenet_anemia.joblib")
    extractor = joblib.load(MODEL_DIR / "mobilenet_extractor.joblib")["model"]
    scaler = classifier_bundle["scaler"]
    classifier = classifier_bundle["classifier"]
    image_size = classifier_bundle["img_size"]
    threshold = classifier_bundle["threshold"]
    OUTPUT_DIR.mkdir(exist_ok=True)

    for path in sorted(INPUT_DIR.iterdir()):
        if path.suffix.lower() not in {".png", ".jpg", ".jpeg"}:
            continue

        image_bgr = cv2.imread(str(path))
        if image_bgr is None:
            continue
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(image_rgb, (image_size, image_size))
        model_input = preprocess_input(resized.astype(np.float32)[None, ...])
        feature_map = extractor.predict(model_input, verbose=0)[0]
        features = feature_map.reshape(1, -1)
        probability = float(classifier.predict_proba(scaler.transform(features))[0, 1])
        predicted_anemia = probability >= threshold

        attention_overlay, _ = extract_features_advanced_masking(
            image_bgr, return_masks=True
        )

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        axes[0].imshow(image_rgb)
        axes[0].set_title("Sebelum: citra asli")
        axes[1].imshow(attention_overlay)
        axes[1].set_title("Sesudah: peta warna jaringan mata")
        for axis in axes:
            axis.axis("off")
        status = "Anemia" if predicted_anemia else "Normal"
        fig.suptitle(f"{path.name} | MobileNetV2: {status}")
        fig.tight_layout()
        destination = OUTPUT_DIR / f"{path.stem}_before_after.png"
        fig.savefig(destination, dpi=160, bbox_inches="tight")
        plt.close(fig)
        print(f"{destination.name}: {status}, p(anemia)={probability:.4f}")


if __name__ == "__main__":
    main()

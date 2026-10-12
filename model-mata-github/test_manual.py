#!/usr/bin/env python3
"""
test_manual.py — Manual sanity-check runner for OUT-OF-DISTRIBUTION eye images.
===============================================================================

Purpose
-------
Run the built pipeline (feature_extraction.py + inference.py + the saved
model_pipeline.joblib) on eye images that are OUTSIDE the training dataset
(India/Italy). This is a SANITY CHECK ONLY, NOT a measurement of accuracy:
these images have no ground-truth Hgb to compare against.

Scope (read-only)
-----------------
- Does NOT modify feature_extraction.py, inference.py, or model_pipeline.joblib.
- No automatic eye detection / cropping (input images are assumed to be
  already cropped close to the conjunctiva, like *forniceal.png / *palpebral.png).
- No accuracy/MAE metrics (no ground truth available).

Gender handling (no silent assumptions)
---------------------------------------
Priority:
  1. Gender pattern detected in the filename (flexible regex).
  2. --gender CLI argument.
  3. Otherwise the Hgb PREDICTION IS SKIPPED for that image:
     mask is still extracted & visualized for visual sanity, and the
     status column is set to "gender_tidak_diketahui".
     There is NO fallback to a guessed "M"/"F".

Usage
-----
    python test_manual.py                          # uses ./test_images, no default gender
    python test_manual.py --gender F               # default gender when filename has no pattern
    python test_manual.py --image-dir somewhere --results-dir out

Output
------
- Per image: test_results/{name}_result.png  (subplot: original | mask)
- test_results/summary.csv                     (one row per image, openable in Excel/Sheets)
- Console: compact table + explicit disclaimer at the end.
"""

import argparse
import csv
import os
import re
import sys
import warnings

warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")  # headless; never plt.show()
import matplotlib.pyplot as plt

import cv2
import numpy as np

from feature_extraction import extract_features_advanced_masking
from inference import predict_from_image

# ---------------------------------------------------------------------------
# Gender detection
# ---------------------------------------------------------------------------
FEMALE_RE = re.compile(r'(_f_|_f\.|_female_|_female\.|_perempuan_|_wanita_|female\d|_p_)')
MALE_RE = re.compile(r'(_m_|_m\.|_male_|_male\.|_pria_|_laki_|male\d)')


def detect_gender_from_filename(filename):
    """Return 'F', 'M' or None (case-insensitive)."""
    fname = filename.lower()
    if FEMALE_RE.search(fname):
        return 'F'
    if MALE_RE.search(fname):
        return 'M'
    return None


# ---------------------------------------------------------------------------
# Sanity flags
# ---------------------------------------------------------------------------
def sanity_flags(valid_ratio, hgb, margin):
    flags = []
    if valid_ratio is not None and valid_ratio < 0.05:
        flags.append("low_mask<5%")
    if hgb is not None and (hgb < 5.0 or hgb > 20.0):
        flags.append("hgb_unphysiologis")
    if margin is not None and abs(margin) < 0.2:
        flags.append("dekat_ambang")
    return flags


# ---------------------------------------------------------------------------
# Image processing
# ---------------------------------------------------------------------------
def process_image(path, gender_cli):
    name = os.path.basename(path)
    row = {
        "file": name,
        "gender_input": "",
        "n_valid_px": None,
        "total_px": None,
        "valid_ratio": None,
        "hgb_predicted": None,
        "status": "",
        "threshold_used": None,
        "margin_ke_threshold": None,
        "flags": [],
        "error": "",
        "img_original": None,
        "img_mask": None,
    }

    img = cv2.imread(path)
    if img is None:
        row["status"] = "error"
        row["error"] = "Gagal membaca file (korup/bukan gambar)"
        return row

    h, w = img.shape[:2]
    row["total_px"] = int(h * w)

    # --- Gender decision (NO silent assumption) ---
    gender = detect_gender_from_filename(name)
    if gender is None and gender_cli:
        gender = gender_cli
    row["gender_input"] = gender if gender else "TIDAK_DIISI"

    # --- Mask + features (n_valid_px from boolean mask, accurate) ---
    # NOTE: n_valid_px comes from the actual boolean mask (final_valid.sum())
    # inside the extractor, NOT from counting non-zero pixels of the
    # zeroed-out segmented image. This avoids the corner case where a
    # valid dark pixel happens to be exactly RGB(0,0,0).
    feats = extract_features_advanced_masking(img, return_masks=False)
    if feats is None:
        masks = None
        row["n_valid_px"] = 0
    else:
        row["n_valid_px"] = int(feats["n_valid_px"])
        masks = extract_features_advanced_masking(img, return_masks=True)

    row["valid_ratio"] = (row["n_valid_px"] / row["total_px"]
                          if row["total_px"] else 0.0)

    if masks is not None:
        _, segmented = masks
        row["img_original"] = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        row["img_mask"] = segmented
    else:
        # Keep original for display; mask panel will show "ROI tidak terdeteksi"
        row["img_original"] = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # --- Prediction (skipped if gender unknown) ---
    if gender is None:
        row["status"] = "gender_tidak_diketahui"
        row["flags"] = ["gender_missing"] + sanity_flags(row["valid_ratio"], None, None)
        return row

    pred = predict_from_image(img, gender)
    if "error" in pred:
        row["status"] = "error"
        row["error"] = pred["error"]
        row["flags"] = sanity_flags(row["valid_ratio"], None, None)
        return row

    row["hgb_predicted"] = pred["hgb_predicted"]
    row["status"] = pred["status"]
    row["threshold_used"] = pred["threshold_used"]
    row["margin_ke_threshold"] = pred["margin_ke_threshold"]
    row["flags"] = sanity_flags(row["valid_ratio"], pred["hgb_predicted"],
                                pred["margin_ke_threshold"])
    return row


# ---------------------------------------------------------------------------
# Visualization (headless)
# ---------------------------------------------------------------------------
def save_figure(row, out_path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].imshow(row["img_original"])
    axes[0].set_title("Original", fontsize=10)
    axes[0].axis("off")

    if row["img_mask"] is not None:
        axes[1].imshow(row["img_mask"])
        axes[1].set_title(f"Mask (valid={row['n_valid_px']}px, "
                          f"{row['valid_ratio']*100:.1f}%)", fontsize=10)
    else:
        axes[1].text(0.5, 0.5, "ROI tidak\ndetected", ha="center", va="center")
        axes[1].set_title("Mask", fontsize=10)
    axes[1].axis("off")

    fig.suptitle(f"{row['file']} | hgb={row['hgb_predicted']} | "
                 f"{row['status']}", fontsize=11)
    plt.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Sanity-check runner for "
                                     "out-of-distribution conjunctival images.")
    parser.add_argument("--image-dir", default="test_images",
                        help="Folder berisi gambar mata ter-crop (default: test_images)")
    parser.add_argument("--gender", default=None,
                        help="Gender default untuk file tanpa pola gender di nama "
                             "(F/M). Jika kosong & tidak ada pola → prediksi di-skip.")
    parser.add_argument("--results-dir", default="test_results",
                        help="Folder hasil (PNG + summary.csv)")
    args = parser.parse_args()

    image_dir = args.image_dir
    results_dir = args.results_dir

    if not os.path.isdir(image_dir):
        print(f"❌ Folder '{image_dir}' tidak ditemukan.")
        print("   Buat foldernya dan taruh gambarnya (jpg/jpeg/png), lalu run lagi.")
        sys.exit(0)

    exts = (".jpg", ".jpeg", ".png")
    files = sorted(f for f in os.listdir(image_dir)
                   if f.lower().endswith(exts) and os.path.isfile(os.path.join(image_dir, f)))

    if not files:
        print(f"ℹ️  Folder '{image_dir}' ada tapi tidak berisi gambar "
              f"({', '.join(exts)}). Belum ada yang diproses.")
        sys.exit(0)

    os.makedirs(results_dir, exist_ok=True)

    rows = []
    print("=" * 100)
    print(f"{'File':<28} {'gender':<10} {'valid_px':>9} {'hgb':>6} {'status':<24} {'margin':>7}  flags")
    print("-" * 100)

    for f in files:
        row = process_image(os.path.join(image_dir, f), args.gender)
        rows.append(row)

        hgb_s = f"{row['hgb_predicted']:.2f}" if row["hgb_predicted"] is not None else "-"
        margin_s = f"{row['margin_ke_threshold']:.2f}" if row["margin_ke_threshold"] is not None else "-"
        status_s = row["status"] or "?"
        if row["error"]:
            status_s = row["error"][:22]
        flags_s = ",".join(row["flags"]) if row["flags"] else "-"
        print(f"{row['file']:<28} {row['gender_input']:<10} {row['n_valid_px']:>9} "
              f"{hgb_s:>6} {status_s:<24} {margin_s:>7}  {flags_s}")

        # Save figure
        stem = os.path.splitext(f)[0]
        save_figure(row, os.path.join(results_dir, f"{stem}_result.png"))

        # Low-mask anomaly note (see module docstring about (0,0,0) pixels)
        if row["valid_ratio"] == 0.0 and row["n_valid_px"] == 0 and row["img_mask"] is not None:
            print(f"     ⚠️  catatan: n_valid_px=0 padahal mask terlihat — cek kemungkinan "
                  f"piksel valid murni RGB(0,0,0) pada area yang ter-segmentasi.")

    # ---- summary.csv ----
    csv_path = os.path.join(results_dir, "summary.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["file", "gender_input", "n_valid_px", "total_px",
                         "valid_ratio", "hgb_predicted", "status",
                         "threshold_used", "margin_ke_threshold",
                         "flag_low_mask", "flag_hgb_unphysiological",
                         "flag_borderline", "flag_gender_missing", "error"])
        for r in rows:
            def has(fname):
                return 1 if fname in r["flags"] else 0
            writer.writerow([
                r["file"], r["gender_input"], r["n_valid_px"], r["total_px"],
                round(r["valid_ratio"], 6) if r["valid_ratio"] is not None else "",
                r["hgb_predicted"] if r["hgb_predicted"] is not None else "",
                r["status"],
                r["threshold_used"] if r["threshold_used"] is not None else "",
                r["margin_ke_threshold"] if r["margin_ke_threshold"] is not None else "",
                has("low_mask<5%"), has("hgb_unphysiologis"),
                has("dekat_ambang"), has("gender_missing"),
                r["error"],
            ])

    print("-" * 100)
    print(f"Selesai: {len(rows)} gambar diproses.")
    print(f"  • Gambar hasil : {results_dir}/  (*_result.png)")
    print(f"  • Ringkasan    : {csv_path}")

    # ---- Explicit disclaimer ----
    print()
    print("⚠️  CATATAN: Hasil testing ini BUKAN pengukuran akurasi. Tidak ada nilai Hgb lab")
    print("pembanding untuk gambar-gambar ini. Gunakan hasil ini HANYA untuk mengecek:")
    print("(1) apakah pipeline berjalan tanpa crash, (2) apakah mask secara visual masuk akal,")
    print("(3) apakah angka prediksi berada di rentang fisiologis wajar.")


if __name__ == "__main__":
    main()
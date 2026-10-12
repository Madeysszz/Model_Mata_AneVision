"""
Shared Conjunctival-Feature Extraction (single source of truth)
================================================================
This module is the *only* place that implements the advanced color-space
masking used to build model features. Both the training pipeline and the
real-time inference module import from here, so the exact same function
(identical thresholds, identical math) is used at training and at
prediction time. Do NOT duplicate this logic anywhere else.

Deployment note: this module intentionally imports only core scientific
libraries (numpy, opencv, scikit-image). It has NO dependency on config.py,
the dataset paths, or any web framework, so it can be imported safely in a
production/inference process without a dataset present.
"""

import cv2
import numpy as np
from skimage.color import rgb2lab, rgb2hsv


def clean_mask_morphology(mask, kernel_size=5):
    """Remove small isolated noise specks and fill small holes in the main blob."""
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    cleaned = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_OPEN, kernel)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel)
    return cleaned.astype(bool)


def keep_largest_component(mask):
    """Keep only the largest connected component (assumption: conjunctiva = the
    largest continuous region)."""
    mask_uint8 = mask.astype(np.uint8)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask_uint8, connectivity=8)
    if num_labels <= 1:
        return mask
    largest_idx = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    return (labels == largest_idx)


def extract_features_advanced_masking(img_bgr, return_masks=False,
                                      min_valid_pixels=100):
    """
    Advanced color-space masking feature extractor.

    Copies verbatim the logic used to build the training features
    (same background rejection, same tissue-mask thresholds, same
    mean/std statistics). Changing ANY value here invalidates the
    deployed model — keep it in sync with training.

    Args:
        img_bgr (np.ndarray): BGR image loaded with OpenCV (cv2.imread)
        return_masks (bool): Whether to return visualization masks
        min_valid_pixels (int): Minimum valid pixel count before the
            tissue mask is relaxed to just the background mask

    Returns:
        dict: feature dictionary, OR
        (heatmap_rgb_ish, segmented_rgb): if return_masks=True, OR
        None: if image has too few valid pixels (< 10)
    """
    # Convert BGR to RGB
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_rgb_f = img_rgb.astype(np.float32) / 255.0

    # Convert to LAB and HSV color spaces
    img_lab = rgb2lab(img_rgb_f)
    img_hsv = rgb2hsv(img_rgb_f)

    # ===========================================================================
    # MASKING LAYER 1: Background Rejection (Physical Background)
    # ===========================================================================
    # Detect white background pixels (> 245 in 0-255 scale == > 0.96 in 0-1)
    white_bg = (img_rgb_f[:, :, 0] > 0.96) & \
               (img_rgb_f[:, :, 1] > 0.96) & \
               (img_rgb_f[:, :, 2] > 0.96)

    # Detect black/dark background pixels (< 10 in 0-255 scale == < 0.04 in 0-1)
    black_bg = (img_rgb_f[:, :, 0] < 0.04) & \
               (img_rgb_f[:, :, 1] < 0.04) & \
               (img_rgb_f[:, :, 2] < 0.04)

    # Valid background = NOT white AND NOT black
    valid_bg = ~(white_bg | black_bg)

    # ===========================================================================
    # MASKING LAYER 2: Biological Tissue Detection
    # ===========================================================================
    # Use LAB 'a' channel (red-green axis) + HSV saturation + LAB lightness
    # Relaxed thresholds to ensure pale/anemic conjunctiva tissues are captured
    tissue_mask = (img_lab[:, :, 1] > 5) & \
                  (img_hsv[:, :, 1] > 0.15) & \
                  (img_lab[:, :, 0] > 15)

    # Final valid region = combination of both masks
    final_valid = valid_bg & tissue_mask

    # Fallback 1: If too few pixels detected, relax mask to just valid background
    if final_valid.sum() < min_valid_pixels:
        final_valid = valid_bg

    # Fallback 2: If still insufficient pixels, reject image entirely
    if final_valid.sum() < 10:
        return None

    # ===========================================================================
    # MASKING LAYER 3: Spatial Cleaning
    # Remove small isolated noise specks (e.g. eyelash/skin artifacts that pass
    # the color thresholds) and keep only the main conjunctiva region.
    # This changes the pixel set used for feature statistics, so it changes the
    # extracted feature VALUES for every image (re-trained accordingly).
    # ===========================================================================
    final_valid_before_layer3 = final_valid.copy()
    final_valid = clean_mask_morphology(final_valid, kernel_size=5)
    final_valid = keep_largest_component(final_valid)

    # Safety: never shrink the valid region below the meaningful threshold;
    # fall back to the pre-Layer-3 mask instead of rejecting an image that
    # previously succeeded.
    if final_valid.sum() < 100:
        final_valid = final_valid_before_layer3

    # ===========================================================================
    # FEATURE EXTRACTION (if not returning masks)
    # ===========================================================================
    if not return_masks:
        # Extract statistical features from valid pixels
        features = {
            "mean_R":  np.mean(img_rgb_f[:, :, 0][final_valid]),
            "mean_G":  np.mean(img_rgb_f[:, :, 1][final_valid]),
            "mean_B":  np.mean(img_rgb_f[:, :, 2][final_valid]),

            "mean_L":  np.mean(img_lab[:, :, 0][final_valid]),
            "mean_a":  np.mean(img_lab[:, :, 1][final_valid]),
            "mean_b":  np.mean(img_lab[:, :, 2][final_valid]),

            "mean_S":  np.mean(img_hsv[:, :, 1][final_valid]),
            "mean_V":  np.mean(img_hsv[:, :, 2][final_valid]),

            "std_a":   np.std(img_lab[:, :, 1][final_valid]),
            "std_b":   np.std(img_lab[:, :, 2][final_valid]),
            "std_S":   np.std(img_hsv[:, :, 1][final_valid]),
            "n_valid_px": final_valid.sum(),
        }

        return features

    # ===========================================================================
    # VISUALIZATION MODE (return masks for plotting)
    # ===========================================================================
    else:
        # Create attention heatmap based on color features
        attention_score = (img_lab[:, :, 1] - img_lab[:, :, 1].min()) * img_hsv[:, :, 1]
        attention_norm = cv2.GaussianBlur(attention_score, (15, 15), 0)
        attention_norm = cv2.normalize(
            attention_norm, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        heatmap = cv2.applyColorMap(attention_norm, cv2.COLORMAP_JET)
        heatmap_rgb = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

        # Apply soft overlay on original image
        attention_overlay = cv2.addWeighted(img_rgb, 0.5, heatmap_rgb, 0.5, 0)

        # Segment image based on tissue mask
        segmented_img = img_rgb.copy()
        segmented_img[~final_valid] = 0

        return attention_overlay, segmented_img
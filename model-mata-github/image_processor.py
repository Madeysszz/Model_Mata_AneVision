"""
Image Feature Extraction Module
===============================
Extracts color features from conjunctival images using advanced masking
"""

import os
import cv2
import numpy as np
import pandas as pd
from config import IMAGE_EXTENSIONS, TARGET_PATTERN
from utils import log_message, progress_bar
from feature_extraction import extract_features_advanced_masking


def find_target_image(folder_path, pattern=None):
    """
    Find target fornixal conjunctiva image in patient folder
    
    Args:
        folder_path (str): Path to patient folder
        pattern (str): Image pattern to match (default uses config)
    
    Returns:
        str: Full path to found image, or None if not found
    """
    if pattern is None:
        pattern = TARGET_PATTERN.replace('*', '').lower()
    
    if not os.path.isdir(folder_path):
        return None
    
    # Search for matching files
    for filename in os.listdir(folder_path):
        filename_lower = filename.lower()
        
        # Check if filename contains pattern
        if pattern in filename_lower:
            # Check if it's a valid image format
            if any(filename.endswith(ext) for ext in ['.png', '.jpg', '.jpeg']):
                return os.path.join(folder_path, filename)
    
    return None


def process_single_patient(patient_row, base_path):
    """
    Process one patient's clinical data and extract image features
    
    Args:
        patient_row (pd.Series): Patient metadata row
        base_path (str): Root dataset path
    
    Returns:
        dict or None: Combined features + metadata, or None if processing failed
    """
    # Get patient identifiers
    patient_id = int(patient_row['Number'])
    country = str(patient_row['country']).strip()
    
    # Construct folder path
    patient_folder = os.path.join(base_path, country, str(patient_id))
    
    # Verify folder exists
    if not os.path.exists(patient_folder):
        log_message = f"Patient folder missing: {patient_folder}"
        return None
    
    # Find target image
    img_path = find_target_image(patient_folder)
    if img_path is None:
        log_message = f"No target image found in: {patient_folder}"
        return None
    
    # Load image
    img_bgr = cv2.imread(img_path)
    if img_bgr is None:
        log_message = f"Failed to load image: {img_path}"
        return None
    
    # Extract features
    features = extract_features_advanced_masking(img_bgr)
    if features is None:
        log_message = f"Feature extraction failed for: {img_path}"
        return None
    
    # Combine features with metadata
    combined = {**features}
    combined['patient_id'] = patient_id
    combined['country'] = country
    combined['hgb'] = patient_row['Hgb']
    combined['gender'] = str(patient_row['Gender'])
    combined['age'] = patient_row['Age']
    combined['image_file'] = os.path.basename(img_path)
    
    return combined


def batch_process_patients(patients_df, base_path, show_progress=True):
    """
    Process all patients in DataFrame sequentially
    
    Args:
        patients_df (pd.DataFrame): Patient metadata DataFrame
        base_path (str): Root dataset path
        show_progress (bool): Display progress bar
    
    Returns:
        pd.DataFrame: Combined features dataframe or empty DF on failure
    """
    total_patients = len(patients_df)
    results = []
    success_count = 0
    skipped_folders = 0
    skipped_images = 0
    skipped_extraction = 0
    
    print(f"\n{'='*60}")
    print("[STEP 2] PROCESSING PATIENT IMAGES")
    print('='*60)
    
    # Loop through all patients
    for idx, row in patients_df.iterrows():
        result = process_single_patient(row, base_path)
        
        if result is not None:
            results.append(result)
            success_count += 1
        else:
            # Determine reason for skip
            patient_folder = os.path.join(base_path, str(row['country']), str(row['Number']))
            if not os.path.exists(patient_folder):
                skipped_folders += 1
            elif find_target_image(patient_folder) is None:
                skipped_images += 1
            else:
                skipped_extraction += 1
        
        # Show progress every N patients
        if show_progress and (idx + 1) % 10 == 0:
            progress_bar(idx + 1, total_patients)
    
    if show_progress:
        print()  # New line after progress bar
    
    # Create DataFrame from results
    df_features = pd.DataFrame(results)
    
    # Summary report
    print(f"\n📊 PROCESSING SUMMARY:")
    print('='*60)
    print(f"   Total patients attempted:     {total_patients}")
    print(f"   ✓ Successful extractions:     {success_count}")
    print(f"   ℹ️  Skipped folders missing:   {skipped_folders}")
    print(f"   ℹ️  Skipped no target image:   {skipped_images}")
    print(f"   ℹ️  Skipped extraction failed: {skipped_extraction}")
    print('='*60)
    
    if success_count == 0:
        raise ValueError("❌ No successful feature extractions!")
    
    return df_features


def visualize_segmentation_demo(df_sample, base_path, save_path=None):
    """
    Create visual demo of segmentation pipeline (sample only)
    
    Args:
        df_sample (pd.DataFrame): Sample dataframe with features
        base_path (str): Dataset root path
        save_path (str): Path to save visualization (optional)
    """
    import matplotlib.pyplot as plt
    
    if len(df_sample) == 0:
        print("No sample data available!")
        return
    
    # Get first sample
    sample = df_sample.iloc[0]
    patient_id = int(sample['patient_id'])
    country = sample['country']
    
    folder_path = os.path.join(base_path, country, str(patient_id))
    img_path = find_target_image(folder_path)
    
    if img_path is None:
        print("Sample image not found!")
        return
    
    # Load and display
    img_bgr = cv2.imread(img_path)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    
    # Generate segmentation visualization
    attention_overlay, segmented_img = extract_features_advanced_masking(
        img_bgr, return_masks=True
    )
    
    # Create figure
    fig, axes = plt.subplots(3, 1, figsize=(6, 12))
    
    axes[0].imshow(img_rgb)
    axes[0].set_title(f"(a) Original Image ({country} #{patient_id})", fontsize=11, fontweight='bold')
    axes[0].axis("off")
    
    axes[1].imshow(attention_overlay)
    axes[1].set_title("(b) Color Attention Map", fontsize=11, fontweight='bold')
    axes[1].axis("off")
    
    axes[2].imshow(segmented_img)
    axes[2].set_title("(c) Segmented Tissue Region", fontsize=11, fontweight='bold')
    axes[2].axis("off")
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"✓ Saved visualization to: {save_path}")
    else:
        plt.show()

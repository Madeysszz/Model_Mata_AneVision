"""
Load Clinical Metadata from Excel Files
========================================
Handles reading and preprocessing of patient demographic data

FIXED VERSION - Resolves pandas compatibility issues
"""

import pandas as pd
import os
from config import BASE_PATH, COUNTRIES
from utils import log_message


def validate_dataset(base_path):
    """
    Check if dataset structure is correct
    
    Args:
        base_path (str): Path to dataset root
    
    Returns:
        tuple: (is_valid: bool, message: str)
    """
    errors = []
    
    # Check country folders
    for country in COUNTRIES:
        country_folder = os.path.join(base_path, country)
        
        if not os.path.exists(country_folder):
            errors.append(f"Country folder missing: {country_folder}")
            continue
        
        # Check for Excel file
        excel_file = os.path.join(country_folder, f"{country}.xlsx")
        if not os.path.exists(excel_file):
            errors.append(f"Excel file missing: {excel_file}")
        
        # Count patient folders
        patient_folders = [d for d in os.listdir(country_folder) 
                         if os.path.isdir(os.path.join(country_folder, d))]
        log_message(f"  {country}: {len(patient_folders)} patient folders found")
    
    if errors:
        return False, "❌ Dataset validation failed:\n" + "\n".join(errors)
    
    return True, "✅ Dataset structure is valid!"


def load_country_data(country_path, country_name):
    """
    Load one country's patient metadata from Excel
    
    Args:
        country_path (str): Path to country folder
        country_name (str): Country name for labeling
    
    Returns:
        pd.DataFrame: Cleaned metadata DataFrame or None on error
    """
    excel_file = os.path.join(country_path, f"{country_name}.xlsx")
    
    try:
        # Read Excel file
        df = pd.read_excel(excel_file)
        
        print(f"\n📊 Loading {country_name} data...")
        print(f"   Source: {excel_file}")
        
        # Check required columns
        required_cols = ['Number', 'Hgb', 'Gender', 'Age']
        available_cols = [col for col in required_cols if col in df.columns]
        
        if len(available_cols) < len(required_cols):
            missing = [c for c in required_cols if c not in available_cols]
            print(f"   ⚠️  Missing columns: {missing}")
            print(f"   Available columns: {list(df.columns)}")
        
        # Keep only essential columns
        cols_needed = [c for c in required_cols if c in df.columns]
        df = df[cols_needed].copy()
        
        # Clean column names (remove extra spaces)
        df.columns = [col.strip() for col in df.columns]
        
        # Handle unnamed/error columns - drop them
        cols_to_drop = [col for col in df.columns if 'Unnamed' in str(col)]
        if cols_to_drop:
            df = df.drop(columns=cols_to_drop)
            print(f"   ✓ Dropped {len(cols_to_drop)} unnamed columns")
        
        # ===========================================================================
        # FIXED: Convert Hgb to numeric (compatible with newer pandas versions)
        # Original code used .replace(',', '.', errors='ignore') which is deprecated
        # New approach: manual replacement using apply()
        # ===========================================================================
        print(f"   🔧 Cleaning Hgb values (replacing commas, removing whitespace)...")
        df['Hgb_cleaned'] = df['Hgb'].astype(str).apply(
            lambda x: x.replace(',', '.').strip() if isinstance(x, str) else x
        )
        df['Hgb'] = pd.to_numeric(df['Hgb_cleaned'], errors='coerce')
        df = df.drop(columns=['Hgb_cleaned'])
        
        # Add country label
        df['country'] = country_name
        
        # Drop rows with missing Hgb values
        initial_count = len(df)
        df = df.dropna(subset=['Hgb'])
        dropped = initial_count - len(df)
        
        if dropped > 0:
            print(f"   ℹ️  Removed {dropped} rows with missing Hgb values")
        
        # Final validation
        final_count = len(df)
        hgb_range = f"{df['Hgb'].min():.1f} – {df['Hgb'].max():.1f}"
        
        print(f"   ✓ Loaded {final_count} valid patients")
        print(f"   • HGB range: {hgb_range} g/dL")
        
        return df
        
    except Exception as e:
        print(f"   ❌ Error loading {country_name} data: {e}")
        log_message(f"Error loading {country_name} data: {e}", "ERROR")
        return None


def load_all_metadata():
    """
    Merge India + Italy datasets into unified patient database
    
    Returns:
        pd.DataFrame: Combined metadata with all patients
    """
    print("\n" + "="*60)
    print("[STEP 1] LOADING CLINICAL METADATA")
    print("="*60)
    
    # Validate dataset structure
    is_valid, msg = validate_dataset(BASE_PATH)
    if not is_valid:
        raise ValueError(msg)
    print(f"   {msg}")
    
    patients_list = []
    
    # Load each country
    for country in COUNTRIES:
        country_path = os.path.join(BASE_PATH, country)
        df = load_country_data(country_path, country)
        
        if df is not None:
            patients_list.append(df)
    
    if not patients_list:
        raise ValueError("No valid data loaded from any country!")
    
    # Concatenate all datasets
    print(f"\n🔗 Merging {len(patients_list)} country datasets...")
    df_meta = pd.concat(patients_list, ignore_index=True)
    
    # Summary statistics
    total_patients = len(df_meta)
    india_count = (df_meta['country'] == 'India').sum()
    italy_count = (df_meta['country'] == 'Italy').sum()
    hgb_min = df_meta['Hgb'].min()
    hgb_max = df_meta['Hgb'].max()
    hgb_mean = df_meta['Hgb'].mean()
    
    print(f"\n{'='*60}")
    print("📈 DATASET SUMMARY:")
    print('='*60)
    print(f"   Total Patients:      {total_patients}")
    print(f"   • India:             {india_count}")
    print(f"   • Italy:             {italy_count}")
    print(f"\n   Hemoglobin Statistics:")
    print(f"   • Minimum:           {hgb_min:.2f} g/dL")
    print(f"   • Maximum:           {hgb_max:.2f} g/dL")
    print(f"   • Mean (avg):        {hgb_mean:.2f} g/dL")
    print('='*60)
    
    return df_meta


def get_patient_statistics(df_meta):
    """
    Calculate summary statistics for metadata DataFrame
    
    Args:
        df_meta (pd.DataFrame): Patient metadata
    
    Returns:
        dict: Dictionary of summary statistics
    """
    stats = {
        'total_patients': len(df_meta),
        'india_count': (df_meta['country'] == 'India').sum(),
        'italy_count': (df_meta['country'] == 'Italy').sum(),
        'hgb_range': (df_meta['Hgb'].min(), df_meta['Hgb'].max()),
        'hgb_mean': df_meta['Hgb'].mean(),
        'hgb_std': df_meta['Hgb'].std(),
    }
    
    # Gender breakdown
    if 'Gender' in df_meta.columns:
        stats['female_count'] = (df_meta['Gender'].str.upper().str.contains('F')).sum()
        stats['male_count'] = (df_meta['Gender'].str.upper().str.contains('M')).sum()
    
    return stats

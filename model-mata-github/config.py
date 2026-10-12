"""
Configuration settings for Anemia Analysis Pipeline
====================================================
Auto-detects dataset path and defines all parameters
"""

import os


def get_dataset_path():
    """
    Auto-detect dataset path based on current environment
    
    Priority order:
    1. EYES_DEFY_ANEMIA_DATASET environment variable
    2. data/dataset anemia inside this repository
    
    Returns:
        str: Valid dataset path
        
    Raises:
        FileNotFoundError: If no valid path found
    """
    project_dir = os.path.dirname(os.path.abspath(__file__))
    possible_paths = [
        os.environ.get("EYES_DEFY_ANEMIA_DATASET"),
        os.path.join(project_dir, "data", "dataset anemia"),
    ]
    
    print("\n" + "="*60)
    print("CONFIGURATION LOADED")
    print("="*60)
    
    for path in possible_paths:
        if path and os.path.exists(path):
            # Normalize path separator
            normalized = os.path.normpath(path)
            print(f"\n✅ Dataset found at: {normalized}")
            return normalized
    
    raise FileNotFoundError(
        "❌ Dataset folder not found in any expected location!\n"
        "\nSet EYES_DEFY_ANEMIA_DATASET or place the dataset in data/dataset anemia."
    )


# ===========================================================================
# DATASET CONFIGURATION
# ===========================================================================
BASE_PATH = get_dataset_path()
COUNTRIES = ["India", "Italy"]


# ===========================================================================
# IMAGE PROCESSING PARAMETERS
# ===========================================================================
IMAGE_EXTENSIONS = ['.png', '.jpg', '.jpeg']
TARGET_PATTERN = '*forniceal*'
MIN_VALID_PIXELS = 100  # Minimum valid pixels before fallback masking


# ===========================================================================
# ANEMIA CLASSIFICATION THRESHOLDS (WHO Criteria)
# ===========================================================================
THRESHOLD_FEMALE = 12.0  # g/dL - Below this indicates anemia for females
THRESHOLD_MALE = 13.0    # g/dL - Below this indicates anemia for males


# ===========================================================================
# MODEL TRAINING CONFIGURATION
# ===========================================================================
RANDOM_STATE = 42  # For reproducible results
TEST_SIZE = 0.2    # 20% of data for testing
CV_FOLDS = 5       # Number of cross-validation folds


# ===========================================================================
# OUTPUT DIRECTORIES
# ===========================================================================
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
PLOTS_DIR = os.path.join(OUTPUT_DIR, "plots")
MODELS_DIR = os.path.join(OUTPUT_DIR, "models")


# Create output directories if they don't exist
for directory in [OUTPUT_DIR, PLOTS_DIR, MODELS_DIR]:
    os.makedirs(directory, exist_ok=True)


# ===========================================================================
# LOGGING CONFIGURATION
# ===========================================================================
LOG_LEVEL = "INFO"
VERBOSE = True

print("\n✓ All configurations loaded successfully!")
print("="*60)

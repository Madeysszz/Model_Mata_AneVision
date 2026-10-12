"""
Utility Functions and Helpers
=============================
Common utilities used across all modules
"""

import os
import glob
from datetime import datetime


def validate_file_exists(path, check_type="file"):
    """
    Safe file/folder existence validation
    
    Args:
        path (str): Path to check
        check_type (str): 'file' or 'directory'
    
    Returns:
        tuple: (exists: bool, message: str)
    """
    if check_type == "file":
        exists = os.path.isfile(path)
        msg_type = "File"
    elif check_type == "directory":
        exists = os.path.isdir(path)
        msg_type = "Directory"
    else:
        return False, f"Invalid check_type: {check_type}"
    
    if exists:
        return True, f"{msg_type} exists: {path}"
    else:
        return False, f"{msg_type} NOT found: {path}"


def get_image_paths(folder_path, pattern="*forniceal*"):
    """
    Find matching image files recursively
    
    Args:
        folder_path (str): Parent folder to search in
        pattern (str): Glob pattern to match (default: '*forniceal*')
    
    Returns:
        list: Full paths of matching images
    """
    extensions = ['*.png', '*.jpg', '*.jpeg']
    all_files = []
    
    for ext in extensions:
        # Search for files with this extension containing the pattern
        search_pattern = os.path.join(folder_path, f"*{pattern}*{ext}")
        found = glob.glob(search_pattern, recursive=False)
        
        # Also try without case sensitivity
        search_pattern_upper = os.path.join(folder_path, f"*{pattern.upper()}*{ext}")
        found += glob.glob(search_pattern_upper, recursive=False)
        
        all_files.extend(found)
    
    return all_files


def ensure_directories(*paths):
    """
    Create output directories if they don't exist
    
    Args:
        *paths: Variable number of directory paths to create
    """
    for path in paths:
        os.makedirs(path, exist_ok=True)
        print(f"  ✓ Directory created/verified: {path}")


def log_message(message, level="INFO"):
    """
    Centralized logging with timestamp
    
    Args:
        message (str): Message to log
        level (str): Log level (INFO, WARNING, ERROR)
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] [{level}] {message}")


def format_number(value, decimals=2):
    """
    Consistent number formatting
    
    Args:
        value (float): Number to format
        decimals (int): Decimal places
    
    Returns:
        str: Formatted string
    """
    if value is None:
        return "N/A"
    return f"{value:.{decimals}f}"


def safe_divide(numerator, denominator, default=0.0):
    """
    Safe division that handles zero division
    
    Args:
        numerator (float): Numerator
        denominator (float): Denominator
        default (float): Default value on division by zero
    
    Returns:
        float: Result or default
    """
    if abs(denominator) < 1e-10:
        return default
    return numerator / denominator


def get_patient_folder_structure(base_path, country_name):
    """
    Get list of patient folders for a specific country
    
    Args:
        base_path (str): Base dataset path
        country_name (str): Country name (India/Italy)
    
    Returns:
        list: List of patient folder paths
    """
    country_path = os.path.join(base_path, country_name)
    
    if not os.path.exists(country_path):
        log_message(f"Country folder not found: {country_path}", "WARNING")
        return []
    
    # Get all subdirectories (patient folders)
    folders = [
        os.path.join(country_path, folder) 
        for folder in os.listdir(country_path)
        if os.path.isdir(os.path.join(country_path, folder))
    ]
    
    return sorted(folders)


def progress_bar(current, total, bar_length=40):
    """
    Simple progress bar display
    
    Args:
        current (int): Current step
        total (int): Total steps
        bar_length (int): Length of progress bar
    """
    percent = float(current) / total * 100
    arrow = '-' * int(percent / 100 * bar_length) + '>'
    spaces = ' ' * int(bar_length - len(arrow))
    
    print(f"\rProgress: [{arrow}{spaces}] {percent:.1f}% ({current}/{total})", end='', flush=True)
    
    if current >= total:
        print()  # New line when complete

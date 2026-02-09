"""
Utility functions for HCC survival prediction project.

This module provides helper functions used across the project.
"""

import os
import json
import pickle
from typing import Any, Dict
from datetime import datetime


def ensure_dir(directory: str) -> None:
    """
    Create directory if it doesn't exist.
    
    Args:
        directory: Path to directory
    """
    os.makedirs(directory, exist_ok=True)


def save_json(data: Dict, filepath: str) -> None:
    """
    Save dictionary to JSON file.
    
    Args:
        data: Dictionary to save
        filepath: Path to output JSON file
    """
    ensure_dir(os.path.dirname(filepath))
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)
    print(f"✓ Saved JSON to: {filepath}")


def load_json(filepath: str) -> Dict:
    """
    Load dictionary from JSON file.
    
    Args:
        filepath: Path to JSON file
        
    Returns:
        Loaded dictionary
    """
    with open(filepath, 'r') as f:
        data = json.load(f)
    return data


def save_pickle(obj: Any, filepath: str) -> None:
    """
    Save object to pickle file.
    
    Args:
        obj: Object to save
        filepath: Path to output pickle file
    """
    ensure_dir(os.path.dirname(filepath))
    with open(filepath, 'wb') as f:
        pickle.dump(obj, f)
    print(f"✓ Saved pickle to: {filepath}")


def load_pickle(filepath: str) -> Any:
    """
    Load object from pickle file.
    
    Args:
        filepath: Path to pickle file
        
    Returns:
        Loaded object
    """
    with open(filepath, 'rb') as f:
        obj = pickle.load(f)
    return obj


def get_timestamp() -> str:
    """
    Get current timestamp as string.
    
    Returns:
        Timestamp in format YYYYMMDD_HHMMSS
    """
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def print_section(title: str, width: int = 60) -> None:
    """
    Print a formatted section header.
    
    Args:
        title: Section title
        width: Width of the header line
    """
    print("\n" + "=" * width)
    print(title.center(width))
    print("=" * width)

"""
Feature engineering utilities for HCC survival prediction.

This module provides functions for creating new features to improve model performance.
"""

import pandas as pd
import numpy as np
from typing import List, Optional


def create_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create interaction features between key clinical variables.
    
    Interactions created:
    - Age * AFP: Combined effect of age and liver stress
    - Age * Albumin: General health status relative to age
    - Hemoglobin * Iron: Anemia/Iron overload interaction
    - AFP / Albumin: Ratio of stress marker to health marker
    
    Args:
        df: Input DataFrame
        
    Returns:
        DataFrame with new interaction features
    """
    df = df.copy()
    
    # Ensure numeric columns are numeric
    numeric_cols = ['Age', 'AFP', 'Albumin', 'Hemoglobin', 'Iron', 'Ferritin']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Age interactions
    if 'Age' in df.columns and 'AFP' in df.columns:
        df['Age_AFP_Interaction'] = df['Age'] * np.log1p(df['AFP'].fillna(0))
    
    if 'Age' in df.columns and 'Albumin' in df.columns:
        df['Age_Albumin_Interaction'] = df['Age'] * df['Albumin']
        
    # Iron panel interactions
    if 'Hemoglobin' in df.columns and 'Iron' in df.columns:
        df['Hemo_Iron_Interaction'] = df['Hemoglobin'] * df['Iron']
        
    # Ratio features
    if 'AFP' in df.columns and 'Albumin' in df.columns:
        # Avoid division by zero
        df['AFP_Albumin_Ratio'] = df['AFP'] / (df['Albumin'].replace(0, 0.01))
        
    print(f"✓ Created interaction features: {[col for col in df.columns if 'Interaction' in col or 'Ratio' in col]}")
    return df


def create_binned_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create binned/discretized features.
    
    Bins created:
    - Age_Group: <30, 30-50, 50-70, >70
    - AFP_Level: Normal (<10), High (10-400), Very High (>400)
    
    Args:
        df: Input DataFrame
        
    Returns:
        DataFrame with new binned features
    """
    df = df.copy()
    
    # Age Binning
    if 'Age' in df.columns:
        df['Age_Group'] = pd.cut(
            df['Age'], 
            bins=[0, 30, 50, 70, 100], 
            labels=['Young', 'Middle', 'Senior', 'Elderly']
        )
        
    # AFP Binning (clinical thresholds often used)
    # Normal: <10-20 ng/mL, Diagnostic: >400 ng/mL
    if 'AFP' in df.columns:
        df['AFP_Level'] = pd.cut(
            df['AFP'],
            bins=[-1, 10, 400, np.inf],
            labels=['Normal', 'Elevated', 'Critical']
        )
        
    print(f"✓ Created binned features: Age_Group, AFP_Level")
    return df


def log_transform_features(df: pd.DataFrame, columns: List[str] = None) -> pd.DataFrame:
    """
    Apply log transformation (log1p) to skewed features.
    
    Args:
        df: Input DataFrame
        columns: List of columns to transform. If None, uses default list.
        
    Returns:
        DataFrame with log-transformed features
    """
    df = df.copy()
    
    if columns is None:
        # Common skewed features in medical data
        columns = ['AFP', 'Bilirubin', 'Creatinine', 'Ferritin', 'AST', 'ALT', 'ALP']
    
    encoded_cols = []
    for col in columns:
        if col in df.columns:
            # Handle non-numeric or missing data
            df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # Create new log feature
            feature_name = f'Log_{col}'
            # Start with 0 for missing/invalid to avoid errors, imputation handles rest
            val = df[col].fillna(0)
            val = val.clip(lower=0) # Ensure no negative values
            df[feature_name] = np.log1p(val)
            encoded_cols.append(feature_name)
            
    print(f"✓ Created log-transformed features: {encoded_cols}")
    return df


def apply_feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply all feature engineering steps.
    
    Args:
        df: Input DataFrame
        
    Returns:
        DataFrame with all new features
    """
    print("\n🛠️ Applying Feature Engineering...")
    
    # 1. Interaction Features
    df = create_interaction_features(df)
    
    # 2. Log Transformations
    df = log_transform_features(df)
    
    # 3. Binning
    df = create_binned_features(df)
    
    print("✓ Feature engineering complete")
    return df

"""
Data preprocessing utilities for HCC survival prediction.

This module provides functions for loading, cleaning, and preprocessing
the Hepatocellular Carcinoma (HCC) dataset.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from typing import Tuple, Dict, Optional


def load_data(filepath: str) -> pd.DataFrame:
    """
    Load HCC dataset from CSV file.
    
    Args:
        filepath: Path to the CSV file containing HCC data
        
    Returns:
        DataFrame containing the HCC dataset
        
    Raises:
        FileNotFoundError: If the specified file does not exist
    """
    try:
        df = pd.read_csv(filepath)
        print(f"✓ Loaded dataset: {len(df)} rows, {len(df.columns)} columns")
        return df
    except FileNotFoundError:
        raise FileNotFoundError(f"Dataset not found at: {filepath}")


def handle_missing_values(
    df: pd.DataFrame, 
    strategy: str = 'median',
    missing_indicator: str = '?'
) -> pd.DataFrame:
    """
    Handle missing values in the dataset.
    
    Args:
        df: Input DataFrame
        strategy: Strategy for handling missing values. Options:
                 - 'median': Fill with median (for numeric columns)
                 - 'mean': Fill with mean (for numeric columns)
                 - 'mode': Fill with mode (for categorical columns)
                 - 'drop': Drop rows with missing values
        missing_indicator: String used to indicate missing values (default: '?')
        
    Returns:
        DataFrame with missing values handled
    """
    df = df.copy()
    
    # Replace missing indicators with NaN
    df = df.replace(missing_indicator, np.nan)
    
    # Count missing values before processing
    missing_before = df.isnull().sum().sum()
    
    if strategy == 'drop':
        df = df.dropna()
    else:
        # Handle numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if strategy == 'median':
            df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
        elif strategy == 'mean':
            df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
        
        # Handle categorical columns with mode
        categorical_cols = df.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            if col != 'Class':  # Don't fill target variable
                mode_value = df[col].mode()
                if len(mode_value) > 0:
                    df[col] = df[col].fillna(mode_value[0])
    
    missing_after = df.isnull().sum().sum()
    print(f"✓ Missing values: {missing_before} → {missing_after} (strategy: {strategy})")
    
    return df


def encode_categorical_features(
    df: pd.DataFrame,
    target_column: str = 'Class'
) -> Tuple[pd.DataFrame, Dict[str, LabelEncoder]]:
    """
    Encode categorical features using LabelEncoder.
    
    Args:
        df: Input DataFrame
        target_column: Name of the target variable column (won't be encoded)
        
    Returns:
        Tuple of:
            - DataFrame with encoded categorical features
            - Dictionary mapping column names to their LabelEncoder objects
    """
    df = df.copy()
    encoders = {}
    
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns
    categorical_cols = [col for col in categorical_cols if col != target_column]
    
    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        encoders[col] = le
    
    print(f"✓ Encoded {len(encoders)} categorical features")
    
    return df, encoders


def encode_target_variable(
    df: pd.DataFrame,
    target_column: str = 'Class'
) -> Tuple[pd.DataFrame, LabelEncoder]:
    """
    Encode the target variable.
    
    Args:
        df: Input DataFrame
        target_column: Name of the target variable column
        
    Returns:
        Tuple of:
            - DataFrame with encoded target variable
            - LabelEncoder object for the target variable
    """
    df = df.copy()
    le = LabelEncoder()
    df[target_column] = le.fit_transform(df[target_column])
    
    print(f"✓ Encoded target variable '{target_column}': {dict(zip(le.classes_, le.transform(le.classes_)))}")
    
    return df, le


def scale_features(
    X_train: pd.DataFrame,
    X_test: Optional[pd.DataFrame] = None
) -> Tuple[np.ndarray, Optional[np.ndarray], StandardScaler]:
    """
    Scale features using StandardScaler.
    
    Args:
        X_train: Training features
        X_test: Test features (optional)
        
    Returns:
        Tuple of:
            - Scaled training features
            - Scaled test features (if provided)
            - Fitted StandardScaler object
    """
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    X_test_scaled = None
    if X_test is not None:
        X_test_scaled = scaler.transform(X_test)
    
    print(f"✓ Scaled features using StandardScaler")
    
    return X_train_scaled, X_test_scaled, scaler


def preprocess_pipeline(
    filepath: str,
    missing_strategy: str = 'median',
    test_size: float = 0.2,
    random_state: int = 42,
    use_feature_engineering: bool = False
) -> Dict:
    """
    Complete preprocessing pipeline for HCC dataset.
    
    Args:
        filepath: Path to raw data CSV
        missing_strategy: Strategy for handling missing values
        test_size: Proportion of data to use for testing
        random_state: Random seed for reproducibility
        use_feature_engineering: Whether to apply feature engineering
        
    Returns:
        Dictionary containing processed data and metadata
    """
    from sklearn.model_selection import train_test_split
    
    print("=" * 60)
    print("HCC DATASET PREPROCESSING PIPELINE")
    print("=" * 60)
    
    # Load data
    df = load_data(filepath)
    
    # Handle missing values
    df = handle_missing_values(df, strategy=missing_strategy)
    
    # Feature Engineering (New Step!)
    if use_feature_engineering:
        from feature_engineering import apply_feature_engineering
        df = apply_feature_engineering(df)
    
    # Encode categorical features
    df, encoders = encode_categorical_features(df)
    
    # Encode target variable
    df, target_encoder = encode_target_variable(df)
    
    # Split features and target
    X = df.drop('Class', axis=1)
    y = df['Class']
    
    feature_names = X.columns.tolist()
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    print(f"✓ Train-test split: {len(X_train)} train, {len(X_test)} test samples")
    
    # Scale features
    X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test)
    
    # Metadata
    metadata = {
        'n_samples': len(df),
        'n_features': len(feature_names),
        'n_train': len(X_train),
        'n_test': len(X_test),
        'class_distribution': y.value_counts().to_dict(),
        'missing_strategy': missing_strategy,
        'test_size': test_size,
        'random_state': random_state,
        'feature_engineering': use_feature_engineering
    }
    
    print("=" * 60)
    print(f"✓ Preprocessing complete!")
    print(f"  - Features: {metadata['n_features']}")
    print(f"  - Train samples: {metadata['n_train']}")
    print(f"  - Test samples: {metadata['n_test']}")
    print("=" * 60)
    
    return {
        'X_train': X_train_scaled,
        'X_test': X_test_scaled,
        'y_train': y_train.values,
        'y_test': y_test.values,
        'feature_names': feature_names,
        'encoders': encoders,
        'target_encoder': target_encoder,
        'scaler': scaler,
        'metadata': metadata
    }

"""
Model training utilities for HCC survival prediction.

This module provides functions for training and saving machine learning models.
"""

from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
import joblib
import numpy as np
from typing import Dict, Any, Optional


def train_random_forest(
    X_train: np.ndarray,
    y_train: np.ndarray,
    hyperparameter_tuning: bool = False,
    class_weight: Optional[str] = None,
    random_state: int = 42
) -> RandomForestClassifier:
    """
    Train Random Forest classifier with optional hyperparameter tuning.
    
    Args:
        X_train: Training features
        y_train: Training labels
        hyperparameter_tuning: Whether to perform grid search CV
        class_weight: Weights associated with classes (e.g., 'balanced')
        random_state: Random seed for reproducibility
        
    Returns:
        Trained Random Forest model
    """
    print("\n🌲 Training Random Forest...")
    
    if hyperparameter_tuning:
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [None, 10, 20, 30],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'class_weight': [class_weight] # Pass through to grid search
        }
        rf = RandomForestClassifier(random_state=random_state)
        grid_search = GridSearchCV(
            rf, param_grid, cv=5, scoring='f1', n_jobs=-1, verbose=1
        )
        grid_search.fit(X_train, y_train)
        print(f"✓ Best parameters: {grid_search.best_params_}")
        print(f"✓ Best CV F1-score: {grid_search.best_score_:.4f}")
        return grid_search.best_estimator_
    else:
        rf = RandomForestClassifier(
            n_estimators=100,
            max_depth=20,
            min_samples_split=5,
            class_weight=class_weight,
            random_state=random_state
        )
        rf.fit(X_train, y_train)
        print(f"✓ Random Forest trained with default parameters (class_weight={class_weight})")
        return rf


def train_decision_tree(
    X_train: np.ndarray,
    y_train: np.ndarray,
    hyperparameter_tuning: bool = False,
    class_weight: Optional[str] = None,
    random_state: int = 42
) -> DecisionTreeClassifier:
    """
    Train Decision Tree classifier with optional hyperparameter tuning.
    
    Args:
        X_train: Training features
        y_train: Training labels
        hyperparameter_tuning: Whether to perform grid search CV
        class_weight: Weights associated with classes (e.g., 'balanced')
        random_state: Random seed for reproducibility
        
    Returns:
        Trained Decision Tree model
    """
    print("\n🌳 Training Decision Tree...")
    
    if hyperparameter_tuning:
        param_grid = {
            'max_depth': [None, 5, 10, 15, 20],
            'min_samples_split': [2, 5, 10, 20],
            'min_samples_leaf': [1, 2, 4, 8],
            'criterion': ['gini', 'entropy'],
            'class_weight': [class_weight] # Pass through
        }
        dt = DecisionTreeClassifier(random_state=random_state)
        grid_search = GridSearchCV(
            dt, param_grid, cv=5, scoring='f1', n_jobs=-1, verbose=1
        )
        grid_search.fit(X_train, y_train)
        print(f"✓ Best parameters: {grid_search.best_params_}")
        print(f"✓ Best CV F1-score: {grid_search.best_score_:.4f}")
        return grid_search.best_estimator_
    else:
        dt = DecisionTreeClassifier(
            max_depth=10,
            min_samples_split=5,
            class_weight=class_weight,
            random_state=random_state
        )
        dt.fit(X_train, y_train)
        print(f"✓ Decision Tree trained with default parameters (class_weight={class_weight})")
        return dt


def train_knn(
    X_train: np.ndarray,
    y_train: np.ndarray,
    hyperparameter_tuning: bool = False
) -> KNeighborsClassifier:
    """
    Train K-Nearest Neighbors classifier with optional hyperparameter tuning.
    
    Args:
        X_train: Training features
        y_train: Training labels
        hyperparameter_tuning: Whether to perform grid search CV
        
    Returns:
        Trained KNN model
    """
    print("\n👥 Training K-Nearest Neighbors...")
    
    if hyperparameter_tuning:
        param_grid = {
            'n_neighbors': [3, 5, 7, 9, 11],
            'weights': ['uniform', 'distance'],
            'metric': ['euclidean', 'manhattan', 'minkowski']
        }
        knn = KNeighborsClassifier()
        grid_search = GridSearchCV(
            knn, param_grid, cv=5, scoring='f1', n_jobs=-1, verbose=1
        )
        grid_search.fit(X_train, y_train)
        print(f"✓ Best parameters: {grid_search.best_params_}")
        print(f"✓ Best CV F1-score: {grid_search.best_score_:.4f}")
        return grid_search.best_estimator_
    else:
        knn = KNeighborsClassifier(n_neighbors=5, weights='uniform')
        knn.fit(X_train, y_train)
        print(f"✓ KNN trained with default parameters")
        return knn


def train_all_models(
    X_train: np.ndarray,
    y_train: np.ndarray,
    hyperparameter_tuning: bool = False,
    class_weight: Optional[str] = None,
    random_state: int = 42
) -> Dict[str, Any]:
    """
    Train all models (Random Forest, Decision Tree, KNN).
    
    Args:
        X_train: Training features
        y_train: Training labels
        hyperparameter_tuning: Whether to perform hyperparameter tuning
        class_weight: Weights associated with classes (e.g., 'balanced')
        random_state: Random seed for reproducibility
        
    Returns:
        Dictionary mapping model names to trained model objects
    """
    print("=" * 60)
    print("TRAINING ALL MODELS")
    print("=" * 60)
    
    models = {
        'Random Forest': train_random_forest(
            X_train, y_train, hyperparameter_tuning, class_weight, random_state
        ),
        'Decision Tree': train_decision_tree(
            X_train, y_train, hyperparameter_tuning, class_weight, random_state
        ),
        'KNN': train_knn(X_train, y_train, hyperparameter_tuning) # KNN doesn't support class_weight in standard implementation
    }
    
    print("\n" + "=" * 60)
    print(f"✓ All {len(models)} models trained successfully!")
    print("=" * 60)
    
    return models


def save_model(model: Any, filepath: str) -> None:
    """
    Save trained model to disk using joblib.
    
    Args:
        model: Trained model object
        filepath: Path where model should be saved
    """
    joblib.dump(model, filepath)
    print(f"✓ Model saved to: {filepath}")


def load_model(filepath: str) -> Any:
    """
    Load trained model from disk.
    
    Args:
        filepath: Path to saved model file
        
    Returns:
        Loaded model object
    """
    model = joblib.load(filepath)
    print(f"✓ Model loaded from: {filepath}")
    return model


def save_all_models(models: Dict[str, Any], output_dir: str = 'models') -> None:
    """
    Save all trained models to disk.
    
    Args:
        models: Dictionary mapping model names to model objects
        output_dir: Directory where models should be saved
    """
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    for name, model in models.items():
        filename = name.lower().replace(' ', '_') + '.pkl'
        filepath = os.path.join(output_dir, filename)
        save_model(model, filepath)

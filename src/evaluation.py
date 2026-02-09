"""
Model evaluation utilities for HCC survival prediction.

This module provides functions for evaluating model performance.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_curve, auc, roc_auc_score
)
from typing import Dict, Any, Optional, Tuple
import os


def evaluate_model(
    model: Any,
    X_test: np.ndarray,
    y_test: np.ndarray,
    model_name: str = "Model"
) -> Dict[str, float]:
    """
    Evaluate a trained model on test data.
    
    Args:
        model: Trained model object
        X_test: Test features
        y_test: Test labels
        model_name: Name of the model for display
        
    Returns:
        Dictionary containing evaluation metrics
    """
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
    
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, average='weighted', zero_division=0),
        'recall': recall_score(y_test, y_pred, average='weighted', zero_division=0),
        'f1_score': f1_score(y_test, y_pred, average='weighted', zero_division=0)
    }
    
    if y_pred_proba is not None:
        metrics['roc_auc'] = roc_auc_score(y_test, y_pred_proba)
    
    print(f"\n📊 {model_name} Performance:")
    print(f"  Accuracy:  {metrics['accuracy']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall:    {metrics['recall']:.4f}")
    print(f"  F1-Score:  {metrics['f1_score']:.4f}")
    if 'roc_auc' in metrics:
        print(f"  ROC AUC:   {metrics['roc_auc']:.4f}")
    
    return metrics


def plot_confusion_matrix(
    model: Any,
    X_test: np.ndarray,
    y_test: np.ndarray,
    model_name: str = "Model",
    class_names: Optional[list] = None,
    save_path: Optional[str] = None
) -> None:
    """
    Plot confusion matrix for a model.
    
    Args:
        model: Trained model object
        X_test: Test features
        y_test: Test labels
        model_name: Name of the model for display
        class_names: List of class names for labels
        save_path: Path to save the figure (optional)
    """
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues',
        xticklabels=class_names or ['Dies', 'Lives'],
        yticklabels=class_names or ['Dies', 'Lives']
    )
    plt.title(f'Confusion Matrix - {model_name}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Confusion matrix saved to: {save_path}")
    
    plt.show()


def plot_roc_curve(
    model: Any,
    X_test: np.ndarray,
    y_test: np.ndarray,
    model_name: str = "Model",
    save_path: Optional[str] = None
) -> None:
    """
    Plot ROC curve for a model.
    
    Args:
        model: Trained model object
        X_test: Test features
        y_test: Test labels
        model_name: Name of the model for display
        save_path: Path to save the figure (optional)
    """
    if not hasattr(model, 'predict_proba'):
        print(f"⚠ {model_name} does not support probability predictions. Skipping ROC curve.")
        return
    
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve - {model_name}')
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ ROC curve saved to: {save_path}")
    
    plt.show()


def print_classification_report(
    model: Any,
    X_test: np.ndarray,
    y_test: np.ndarray,
    model_name: str = "Model",
    class_names: Optional[list] = None
) -> None:
    """
    Print detailed classification report.
    
    Args:
        model: Trained model object
        X_test: Test features
        y_test: Test labels
        model_name: Name of the model for display
        class_names: List of class names for labels
    """
    y_pred = model.predict(X_test)
    
    print(f"\n📋 Classification Report - {model_name}")
    print("=" * 60)
    print(classification_report(
        y_test, y_pred,
        target_names=class_names or ['Dies', 'Lives'],
        zero_division=0
    ))


def evaluate_all_models(
    models: Dict[str, Any],
    X_test: np.ndarray,
    y_test: np.ndarray,
    class_names: Optional[list] = None,
    save_dir: Optional[str] = None
) -> pd.DataFrame:
    """
    Evaluate all models and create comparison table.
    
    Args:
        models: Dictionary mapping model names to model objects
        X_test: Test features
        y_test: Test labels
        class_names: List of class names for labels
        save_dir: Directory to save evaluation figures
        
    Returns:
        DataFrame containing comparison of all models
    """
    print("\n" + "=" * 60)
    print("EVALUATING ALL MODELS")
    print("=" * 60)
    
    results = []
    
    for name, model in models.items():
        # Evaluate model
        metrics = evaluate_model(model, X_test, y_test, name)
        metrics['model'] = name
        results.append(metrics)
        
        # Print classification report
        print_classification_report(model, X_test, y_test, name, class_names)
        
        # Plot confusion matrix
        if save_dir:
            cm_path = os.path.join(save_dir, 'figures', f'confusion_matrix_{name.lower().replace(" ", "_")}.png')
            plot_confusion_matrix(model, X_test, y_test, name, class_names, cm_path)
        else:
            plot_confusion_matrix(model, X_test, y_test, name, class_names)
        
        # Plot ROC curve
        if save_dir:
            roc_path = os.path.join(save_dir, 'figures', f'roc_curve_{name.lower().replace(" ", "_")}.png')
            plot_roc_curve(model, X_test, y_test, name, roc_path)
        else:
            plot_roc_curve(model, X_test, y_test, name)
    
    # Create comparison DataFrame
    results_df = pd.DataFrame(results)
    results_df = results_df[['model', 'accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']]
    results_df = results_df.sort_values('f1_score', ascending=False)
    
    print("\n" + "=" * 60)
    print("MODEL COMPARISON")
    print("=" * 60)
    print(results_df.to_string(index=False))
    
    # Save results
    if save_dir:
        results_path = os.path.join(save_dir, 'metrics', 'model_comparison.csv')
        os.makedirs(os.path.dirname(results_path), exist_ok=True)
        results_df.to_csv(results_path, index=False)
        print(f"\n✓ Results saved to: {results_path}")
    
    return results_df


def plot_model_comparison(
    results_df: pd.DataFrame,
    save_path: Optional[str] = None
) -> None:
    """
    Create bar chart comparing model performance.
    
    Args:
        results_df: DataFrame containing model comparison results
        save_path: Path to save the figure (optional)
    """
    metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']
    metrics = [m for m in metrics if m in results_df.columns]
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    x = np.arange(len(results_df))
    width = 0.15
    
    for i, metric in enumerate(metrics):
        offset = width * (i - len(metrics) / 2)
        ax.bar(x + offset, results_df[metric], width, label=metric.replace('_', ' ').title())
    
    ax.set_xlabel('Model')
    ax.set_ylabel('Score')
    ax.set_title('Model Performance Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(results_df['model'])
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim([0, 1.1])
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Comparison chart saved to: {save_path}")
    
    plt.show()

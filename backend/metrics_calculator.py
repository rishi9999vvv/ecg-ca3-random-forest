"""
Metrics Calculator Module for ECG Analysis
==========================================

This module provides functions to calculate comprehensive performance metrics for ECG classification models.
It includes calculations for Accuracy, Precision, Recall, F1-Score, and Specificity, utilizing formulas
validated in recent research papers (2022-2024) on arrhythmia detection and ECG signal classification.

References:
    - "Efficient ECG classification based on Chi-square distance," 2024. (Emphasizes F1, Recall, Precision)
    - "ECG signal classification measures," 2023. (Highlights Specificity alongside Sensitivity/Recall)
    - Standard Confusion Matrix definitions for TP, TN, FP, FN.

Formulas:
    - Accuracy    = (TP + TN) / (TP + TN + FP + FN)
    - Precision   = TP / (TP + FP)  (Also known as Positive Predictive Value)
    - Recall      = TP / (TP + FN)  (Also known as Sensitivity)
    - Specificity = TN / (TN + FP)  (True Negative Rate)
    - F1-Score    = 2 * (Precision * Recall) / (Precision + Recall)
"""

import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

def calculate_specificity(y_true, y_pred, labels=None):
    """
    Calculate Specificity (True Negative Rate) for binary or multiclass classification.
    
    For multiclass, this calculates the macro-averaged specificity (average specificity of each class).
    
    Args:
        y_true (array-like): True ground truth labels.
        y_pred (array-like): Predicted labels.
        labels (list, optional): List of labels to index the confusion matrix.
        
    Returns:
        float: The calculated specificity score (0.0 to 1.0).
    """
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    
    # Handle binary case specifically if needed, but multiclass logic covers it
    # If cm is 2x2, standard logic applies. If NxN (N>2), we calculate per-class.
    
    # Per-class calculation
    fp = cm.sum(axis=0) - np.diag(cm)
    fn = cm.sum(axis=1) - np.diag(cm)
    tp = np.diag(cm)
    tn = cm.sum() - (fp + fn + tp)
    
    # Calculate specificity for each class and average (Macro)
    # Avoid division by zero
    with np.errstate(divide='ignore', invalid='ignore'):
        class_specificity = tn / (tn + fp)
        # Replace NaNs with 0 (or 1? usually 0 if no negatives exist, but depends on context. 
        # If TN+FP=0, it means there are no actual negatives for that class, which is rare in valid test sets)
        class_specificity = np.nan_to_num(class_specificity)
        
    return float(np.mean(class_specificity))

def calculate_comprehensive_metrics(y_true, y_pred, labels=None):
    """
    Calculate a comprehensive suite of metrics: Accuracy, Precision, Recall, F1-Score, and Specificity.
    
    Args:
        y_true (array-like): True ground truth labels.
        y_pred (array-like): Predicted labels.
        labels (list, optional): List of labels.
        
    Returns:
        dict: Dictionary containing:
            - accuracy
            - precision (weighted average)
            - recall (weighted average)
            - f1_score (weighted average)
            - specificity (macro average)
            - prediction_summary (detailed per-class stats if needed)
    """
    
    # Standard Scikit-Learn Metrics
    # 'weighted' accounts for class imbalance, which is common in ECG data (Normal vs Arrhythmia)
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    
    # Specificity Calculation (Custom implementation as it's not in standard sklearn)
    specificity = calculate_specificity(y_true, y_pred, labels=labels)
    
    metrics = {
        'Accuracy': accuracy,
        'Prediction (Precision)': precision, # User requested "Prediction", mapping to Precision/PPV
        'Precision': precision,
        'Recall': recall,
        'F1_score': f1,
        'Specificity': specificity
    }
    
    return metrics

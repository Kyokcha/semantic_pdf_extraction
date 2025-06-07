"""Model evaluation utilities for classification models."""

import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np


def evaluate_model(clf, X_test, y_test, y_pred, class_labels=None) -> None:
    """Evaluate a classifier and print performance metrics.
    
    Args:
        clf (sklearn.base.ClassifierMixin): Fitted classifier object with predict method.
        X_test (pd.DataFrame): Test feature matrix.
        y_test (array-like): Ground truth labels.
        y_pred (array-like): Predicted labels from the model.
        class_labels (list, optional): Class label names. If None, inferred from y_test.
    
    Note:
        Feature importance display is limited to top 20 features.
        Only works with classifiers that have feature_importances_ attribute.
    """
    print("\nClassification Report:\n")
    print(classification_report(y_test, y_pred))

    print("Confusion Matrix:")

    # If class_labels is None, infer from y_test
    labels = class_labels if class_labels is not None else np.unique(y_test)
    cm = confusion_matrix(y_test, y_pred, labels=labels)

    # Try to name the rows/columns clearly
    label_names = class_labels if isinstance(class_labels, (list, np.ndarray)) else labels
    cm_df = pd.DataFrame(cm, index=label_names, columns=label_names)
    print(cm_df.to_string())

    # Show feature importances
    if hasattr(clf, "feature_importances_"):
        print("\nTop Feature Importances:")
        importances = pd.Series(clf.feature_importances_, index=X_test.columns)
        print(importances.sort_values(ascending=False).head(20))
"""LightGBM classifier training module."""

from lightgbm import LGBMClassifier


def train_lightgbm(X_train, y_train, random_state) -> LGBMClassifier:
    """Train a LightGBM classifier for multiclass classification.
    
    Args:
        X_train (pd.DataFrame): Training feature matrix.
        y_train (array-like): Training labels.
        random_state (int): Random seed for reproducibility.
    
    Returns:
        LGBMClassifier: Trained LightGBM classifier.
    """
    clf = LGBMClassifier(
        objective='multiclass',
        random_state=random_state
    )
    clf.fit(X_train, y_train)
    return clf
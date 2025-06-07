"""Logistic Regression classifier training module."""

from sklearn.linear_model import LogisticRegression


def train_logistic_regression(X_train, y_train, random_state) -> LogisticRegression:
    """Train a Logistic Regression classifier.
    
    Args:
        X_train (pd.DataFrame): Training feature matrix.
        y_train (array-like): Training labels.
        random_state (int): Random seed for reproducibility.
    
    Returns:
        LogisticRegression: Trained Logistic Regression classifier.
        
    Note:
        Uses max_iter=1000 to ensure convergence with complex datasets.
    """
    clf = LogisticRegression(max_iter=1000, random_state=random_state)
    clf.fit(X_train, y_train)
    return clf
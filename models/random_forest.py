"""Random Forest classifier training module."""

from sklearn.ensemble import RandomForestClassifier


def train_random_forest(X_train, y_train, random_state) -> RandomForestClassifier:
    """Train a Random Forest classifier.
    
    Args:
        X_train (pd.DataFrame): Training feature matrix.
        y_train (array-like): Training labels.
        random_state (int): Random seed for reproducibility.
    
    Returns:
        RandomForestClassifier: Trained Random Forest classifier.
        
    Note:
        Uses 100 trees (n_estimators=100) as default ensemble size.
    """
    clf = RandomForestClassifier(n_estimators=100, random_state=random_state)
    clf.fit(X_train, y_train)
    return clf
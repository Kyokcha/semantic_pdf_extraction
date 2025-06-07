"""XGBoost classifier training module."""

from xgboost import XGBClassifier


def train_xgboost(X_train, y_train, random_state) -> XGBClassifier:
    """Train an XGBoost classifier for multiclass classification.
    
    Args:
        X_train (pd.DataFrame): Training feature matrix.
        y_train (array-like): Training labels.
        random_state (int): Random seed for reproducibility.
    
    Returns:
        XGBClassifier: Trained XGBoost classifier.
        
    Note:
        Uses 'multi:softprob' objective for multiclass probability output.
        Label encoder is disabled to prevent deprecation warnings.
    """
    clf = XGBClassifier(
        objective='multi:softprob',
        eval_metric='mlogloss',
        use_label_encoder=False,
        random_state=random_state
    )
    clf.fit(X_train, y_train)
    return clf
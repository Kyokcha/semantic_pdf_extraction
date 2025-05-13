# models/xgboost_model.py

from xgboost import XGBClassifier


def train_xgboost(X_train, y_train, random_state):
    clf = XGBClassifier(
        objective='multi:softprob',
        eval_metric='mlogloss',
        use_label_encoder=False,
        random_state=random_state
    )
    clf.fit(X_train, y_train)
    return clf

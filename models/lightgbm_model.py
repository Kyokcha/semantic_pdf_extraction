# models/lightgbm_model.py

from lightgbm import LGBMClassifier


def train_lightgbm(X_train, y_train, random_state):
    clf = LGBMClassifier(
        objective='multiclass',
        random_state=random_state
    )
    clf.fit(X_train, y_train)
    return clf

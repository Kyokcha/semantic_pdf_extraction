# models/logistic_regression.py

from sklearn.linear_model import LogisticRegression

def train_logistic_regression(X_train, y_train, random_state):
    clf = LogisticRegression(max_iter=1000, random_state=random_state)
    clf.fit(X_train, y_train)
    return clf

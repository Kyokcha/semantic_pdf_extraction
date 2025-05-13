# models/random_forest.py

from sklearn.ensemble import RandomForestClassifier

def train_random_forest(X_train, y_train, random_state):
    clf = RandomForestClassifier(n_estimators=100, random_state=random_state)
    clf.fit(X_train, y_train)
    return clf
# models/base.py

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix


def evaluate_model(clf, X_test, y_test, y_pred):
    print("\nClassification Report:\n")
    print(classification_report(y_test, y_pred))

    print("Confusion Matrix:")
    cm = confusion_matrix(y_test, y_pred, labels=clf.classes_)
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=clf.classes_, yticklabels=clf.classes_)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.show()

    # Show feature importances for models that support it
    if hasattr(clf, "feature_importances_"):
        print("\nTop Feature Importances:")
        importances = pd.Series(clf.feature_importances_, index=X_test.columns)
        print(importances.sort_values(ascending=False).head(20))
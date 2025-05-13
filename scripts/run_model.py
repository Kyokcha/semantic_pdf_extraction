# scripts/run_model.py

import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from utils.config import load_config

from models.base import evaluate_model
from models.random_forest import train_random_forest
from models.logistic_regression import train_logistic_regression
from models.xgboost_model import train_xgboost
from models.lightgbm_model import train_lightgbm


def train_model(data_path: str, target_col: str = 'best_extractor', test_size: float = 0.2, random_state: int = 42):
    """
    Train the selected model using document-level splitting.

    This ensures that all sentences from the same article are either in the training
    set or the test set — not both — so we can properly reconstruct full documents.
    """
    config = load_config()
    model_flags = config["model_config"]["model_selection"]

    selected_models = [k for k, v in model_flags.items() if v is True]
    if len(selected_models) != 1:
        raise ValueError(f"Expected exactly one model to be selected in config, found: {selected_models}")

    model_type = selected_models[0]

    model_map = {
        'random_forest': train_random_forest,
        'logistic_regression': train_logistic_regression,
        'xgboost': train_xgboost,
        'lightgbm': train_lightgbm,
    }

    df = pd.read_csv(data_path)

    # Extract article ID
    df["article_id"] = df["gt_sentence_id"].str.extract(r"^(article_\d{3})")[0]

    # Optionally filter out ties
    # df = df[df['is_tie'] == False]

    # Document-level split
    unique_articles = df["article_id"].unique()
    train_articles, test_articles = train_test_split(
        unique_articles, test_size=test_size, random_state=random_state
    )

    train_df = df[df["article_id"].isin(train_articles)].reset_index(drop=True)
    test_df = df[df["article_id"].isin(test_articles)].reset_index(drop=True)

    drop_cols = [target_col, 'article_id', 'gt_sentence_id', 'is_tie',
                 'sentence_pypdf2', 'sentence_ocr', 'sentence_plumber']

    X_train = train_df.drop(columns=drop_cols).reset_index(drop=True)
    y_train = train_df[target_col].reset_index(drop=True)
    X_test = test_df.drop(columns=drop_cols).reset_index(drop=True)
    y_test = test_df[target_col].reset_index(drop=True)

    clf = model_map[model_type](X_train, y_train, random_state)
    y_pred = clf.predict(X_test)

    evaluate_model(clf, X_test, y_test, y_pred)

    # Save model predictions
    output_dir = Path(config["data_paths"]["model_outputs"])
    output_dir.mkdir(parents=True, exist_ok=True)

    X_test_with_meta = X_test.copy()
    X_test_with_meta["gt_sentence_id"] = test_df["gt_sentence_id"]
    X_test_with_meta["article_id"] = test_df["article_id"]
    X_test_with_meta["predicted_extractor"] = y_pred

    X_test_with_meta["selected_sentence"] = [
        test_df.loc[i, f"sentence_{extractor}"] for i, extractor in zip(test_df.index, y_pred)
    ]

    X_test_with_meta[["article_id", "gt_sentence_id", "predicted_extractor", "selected_sentence"]] \
        .to_csv(output_dir / "model_predictions.csv", index=False)

    print(f"✅ Saved model predictions to {output_dir / 'model_predictions.csv'}")


if __name__ == "__main__":
    config = load_config()
    data_flags = config["model_config"]["data_selection"]
    data_paths = config["model_config"]["data_paths"]

    selected_data = [k for k, v in data_flags.items() if v is True]
    if len(selected_data) != 1:
        raise ValueError(f"Expected exactly one dataset to be selected in config, found: {selected_data}")

    selected_key = selected_data[0]
    if selected_key not in data_paths:
        raise ValueError(f"No path configured for dataset key: {selected_key}")

    train_model(data_path=data_paths[selected_key])

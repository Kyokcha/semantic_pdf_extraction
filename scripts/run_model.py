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

    # Add article_id back if needed (assuming your gt_sentence_id still encodes it)
    df["article_id"] = df["gt_sentence_id"].str.extract(r"^(doc_\d{3})")[0]

    df = df[df['is_tie'] == False]  # filter out ambiguous cases

    # Split by document
    train_ids, test_ids = train_test_split(df["article_id"].unique(), test_size=test_size, random_state=random_state)
    train_df = df[df["article_id"].isin(train_ids)]
    test_df = df[df["article_id"].isin(test_ids)]

    # Prepare features
    drop_cols = ['article_id', 'gt_sentence_id', 'best_extractor', 'is_tie',
                 'sentence_pypdf2', 'sentence_ocr', 'sentence_plumber', 'similarity_score_pypdf2',
                 'similarity_score_ocr', 'similarity_score_plumber']
    X_train = train_df.drop(columns=drop_cols)
    y_train = train_df['best_extractor']
    X_test = test_df.drop(columns=drop_cols)
    y_test = test_df['best_extractor']

    clf = model_map[model_type](X_train, y_train, random_state)
    y_pred = clf.predict(X_test)

    evaluate_model(clf, X_test, y_test, y_pred)

    # Save predictions
    output_dir = Path(config["data_paths"]["DB_model_outputs"])
    output_dir.mkdir(parents=True, exist_ok=True)

    X_test_with_meta = X_test.copy()
    X_test_with_meta["gt_sentence_id"] = test_df["gt_sentence_id"].values
    X_test_with_meta["article_id"] = test_df["article_id"].values
    X_test_with_meta["predicted_extractor"] = y_pred
    X_test_with_meta["selected_sentence"] = [
        test_df.iloc[i][f"sentence_{extractor}"] for i, extractor in enumerate(y_pred)
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

"""Train and evaluate multiple classifier models for extractor selection."""

import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from utils.config import load_config
from models.base import evaluate_model
from models.random_forest import train_random_forest
from models.xgboost_model import train_xgboost
from models.lightgbm_model import train_lightgbm
import pickle


def run_model_variant(name: str, train_func, X_train: pd.DataFrame, X_test: pd.DataFrame, 
                     y_train: pd.Series, y_test: pd.Series, test_df: pd.DataFrame, 
                     output_dir: Path) -> tuple:
    """Train and evaluate a specific model variant.
    
    Args:
        name (str): Name identifier for the model variant.
        train_func (callable): Model training function.
        X_train (pd.DataFrame): Training features.
        X_test (pd.DataFrame): Test features.
        y_train (pd.Series): Training labels.
        y_test (pd.Series): Test labels.
        test_df (pd.DataFrame): Full test dataset with metadata.
        output_dir (Path): Directory to save model predictions.
    
    Returns:
        tuple: (trained_model, label_encoder, test_accuracy, model_name)
    """
    print(f"\n⚙️ Training and evaluating {name}...")

    le = LabelEncoder()
    y_train_enc = le.fit_transform(y_train)
    y_test_enc = le.transform(y_test)

    clf = train_func(X_train, y_train_enc, random_state=42)
    y_pred_enc = clf.predict(X_test)

    y_test_labels = le.inverse_transform(y_test_enc)
    y_pred_labels = le.inverse_transform(y_pred_enc)

    metrics = evaluate_model(clf, X_test, y_test_labels, y_pred_labels, class_labels=le.classes_)

    output_path = output_dir / f"model_predictions_{name}.csv"
    X_test_with_meta = X_test.copy()
    X_test_with_meta["gt_sentence_id"] = test_df["gt_sentence_id"].values
    X_test_with_meta["article_id"] = test_df["article_id"].values
    X_test_with_meta["predicted_extractor"] = y_pred_labels
    X_test_with_meta["selected_sentence"] = [
        test_df.iloc[i][f"sentence_{label}"] for i, label in enumerate(y_pred_labels)
    ]
    X_test_with_meta[["article_id", "gt_sentence_id", "predicted_extractor", "selected_sentence"]] \
        .to_csv(output_path, index=False)

    print(f"✅ Saved predictions for {name} to {output_path}")
    
    return clf, le, metrics['accuracy'], name


def main() -> None:
    """Train all model variants on the selected dataset.
    
    Loads data based on configuration, splits into train/test sets,
    and evaluates multiple classifier models. Saves the best performing model.
    
    Note:
        Requires exactly one dataset to be selected in config.
        Uses 80/20 train/test split at article level.
        Skips tie cases where best extractor is ambiguous.
    """
    config = load_config()
    data_flags = config["model_config"]["data_selection"]
    data_paths = config["model_config"]["data_paths"]

    selected_data = [k for k, v in data_flags.items() if v is True]
    if len(selected_data) != 1:
        raise ValueError(f"Expected exactly one dataset to be selected in config, found: {selected_data}")

    selected_key = selected_data[0]
    if selected_key not in data_paths:
        raise ValueError(f"No path configured for dataset key: {selected_key}")

    data_path = data_paths[selected_key]
    df = pd.read_csv(data_path)
    df["article_id"] = df["gt_sentence_id"].str.extract(r"^(doc_\d{3})")[0]
    df = df[df['is_tie'] == False]

    train_ids, test_ids = train_test_split(df["article_id"].unique(), test_size=0.2, random_state=42)
    train_df = df[df["article_id"].isin(train_ids)]
    test_df = df[df["article_id"].isin(test_ids)]

    drop_cols = ['article_id', 'gt_sentence_id', 'best_extractor', 'is_tie',
                 'sentence_pypdf2', 'sentence_ocr', 'sentence_plumber',
                 'similarity_score_pypdf2', 'similarity_score_ocr', 'similarity_score_plumber',
                 'cosine_sim_with_ocr_ocr','cosine_sim_with_pypdf2_pypdf2','cosine_sim_with_plumber_plumber',
                 'jaccard_with_ocr_ocr','jaccard_with_pypdf2_pypdf2','jaccard_with_plumber_plumber']

    X_train = train_df.drop(columns=drop_cols)
    y_train = train_df['best_extractor']
    X_test = test_df.drop(columns=drop_cols)
    y_test = test_df['best_extractor']

    output_dir = Path(config["data_paths"]["DB_model_outputs"])
    output_dir.mkdir(parents=True, exist_ok=True)

    models = {
        'random_forest': train_random_forest,
        'xgboost': train_xgboost,
        'lightgbm': train_lightgbm,
    }

    # Train all models and track their performance
    model_results = []
    for name, func in models.items():
        model_result = run_model_variant(name, func, X_train, X_test, y_train, y_test, test_df, output_dir)
        model_results.append(model_result)

    # Find the best performing model
    best_model, best_encoder, best_accuracy, best_name = max(model_results, key=lambda x: x[2])

    # Save the best model and its metadata
    model_info = {
        'model': best_model,
        'encoder': best_encoder,
        'feature_columns': X_train.columns.tolist()
    }
    
    with open(output_dir / "best_model.pkl", 'wb') as f:
        pickle.dump(model_info, f)
    
    print(f"\n📋 Model Performance Summary:")
    for _, _, accuracy, name in model_results:
        print(f"{name}: {accuracy:.3f}")
    
    print(f"\n🏆 Best model ({best_name}, accuracy: {best_accuracy:.3f}) has been pickled for future use.")


if __name__ == "__main__":
    main()
"""Evaluate model performance using semantic similarity scores and statistical tests."""

import pandas as pd
from pathlib import Path
from utils.config import load_config
import logging
from scipy.stats import ttest_ind
import numpy as np

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_predicted_score(row: pd.Series) -> float:
    """Get similarity score for the predicted extractor.
    
    Args:
        row (pd.Series): DataFrame row containing predictions and scores.
    
    Returns:
        float: Similarity score for the predicted extractor.
    """
    key = f"similarity_score_{row['predicted_extractor']}"
    return row.get(key, None)


def get_best_extractor(row: pd.Series) -> str:
    """Determine best extractor based on similarity scores.
    
    Args:
        row (pd.Series): DataFrame row containing similarity scores.
    
    Returns:
        str: Name of extractor with highest similarity score.
    """
    scores = {
        "pypdf2": row["similarity_score_pypdf2"],
        "ocr": row["similarity_score_ocr"],
        "plumber": row["similarity_score_plumber"]
    }
    return max(scores, key=scores.get)


def compare_extractors(model_scores: pd.Series, other_scores: pd.Series, 
                      other_name: str) -> dict:
    """Compare model scores against another extractor using t-test.
    
    Args:
        model_scores (pd.Series): Similarity scores from model predictions.
        other_scores (pd.Series): Similarity scores from comparison extractor.
        other_name (str): Name of comparison extractor.
    
    Returns:
        dict: Comparison results including means, t-stat, p-value and interpretation.
    """
    t_stat, p_val = ttest_ind(model_scores, other_scores, equal_var=False)
    model_mean = model_scores.mean()
    other_mean = other_scores.mean()
    if p_val < 0.05:
        if model_mean > other_mean:
            result = f"Model significantly better than {other_name}"
        else:
            result = f"{other_name} significantly better than Model"
    else:
        result = f"Tie with {other_name}"
    return {
        "extractor": other_name,
        "model_mean": model_mean,
        "other_mean": other_mean,
        "t_statistic": t_stat,
        "p_value": p_val,
        "comparison_result": result
    }


def main() -> None:
    """Evaluate model performance through statistical comparison of similarity scores.
    
    Performs both document-level and aggregate-level t-tests comparing model
    predictions against individual extractors.
    
    Note:
        Uses Welch's t-test (unequal variance) for all comparisons.
        Significance level is set at p < 0.05.
        Outputs per-document comparisons to CSV and prints aggregate results.
    """
    config = load_config()

    pred_path = Path(config["data_paths"]["DB_model_outputs"]) / "model_predictions.csv"
    train_path = Path(config["data_paths"]["DB_final_training_data"]) / "training_data.csv"
    output_dir = Path(config["data_paths"]["DB_evaluation_outputs"])
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / "semantic_score_comparison.csv"

    # Load data
    logger.info("Loading model predictions and training data...")
    preds = pd.read_csv(pred_path)
    train = pd.read_csv(train_path)

    # Deduplicate training data in case of tie duplicates
    train = train.drop_duplicates(subset=["gt_sentence_id"], keep="first")

    # Merge on gt_sentence_id
    merged = preds.merge(train, on="gt_sentence_id", how="left")

    # Select relevant columns
    cols = [
        "article_id", "gt_sentence_id", "predicted_extractor",
        "similarity_score_pypdf2", "similarity_score_ocr", "similarity_score_plumber"
    ]
    data = merged[cols].copy()

    # Compute scores and best extractors
    data["predicted_similarity"] = data.apply(get_predicted_score, axis=1)
    data["best_extractor"] = data.apply(get_best_extractor, axis=1)

    # Per-document t-tests
    results = []

    for article_id, group in data.groupby("article_id"):
        predicted_scores = group["predicted_similarity"].dropna()
        best_extractor = group["best_extractor"].mode()[0]
        best_scores = group[f"similarity_score_{best_extractor}"].dropna()

        if len(predicted_scores) > 1 and len(best_scores) > 1:
            t_stat, p_val = ttest_ind(predicted_scores, best_scores, equal_var=False)
        else:
            t_stat, p_val = np.nan, np.nan

        if pd.isna(p_val):
            result = "Insufficient data"
        elif p_val < 0.05:
            if predicted_scores.mean() > best_scores.mean():
                result = "Model significantly better"
            else:
                result = "Best extractor significantly better"
        else:
            result = "Tie"

        results.append({
            "article_id": article_id,
            "predicted_similarity": predicted_scores.mean(),
            "similarity_score_pypdf2": group["similarity_score_pypdf2"].mean(),
            "similarity_score_ocr": group["similarity_score_ocr"].mean(),
            "similarity_score_plumber": group["similarity_score_plumber"].mean(),
            "best_extractor": best_extractor,
            "t_statistic": t_stat,
            "p_value": p_val,
            "comparison_result": result
        })

    summary = pd.DataFrame(results)

    # Save per-document comparison
    summary.to_csv(output_path, index=False)
    logger.info(f"✓ Saved semantic similarity comparison to {output_path}")

    # === Aggregate-level t-tests across all sentences ===
    logger.info("Running aggregate-level t-tests...")

    all_model_scores = data['predicted_similarity'].dropna()
    all_pypdf2_scores = data['similarity_score_pypdf2'].dropna()
    all_ocr_scores = data['similarity_score_ocr'].dropna()
    all_plumber_scores = data['similarity_score_plumber'].dropna()

    aggregate_results = [
        compare_extractors(all_model_scores, all_pypdf2_scores, "pypdf2"),
        compare_extractors(all_model_scores, all_ocr_scores, "ocr"),
        compare_extractors(all_model_scores, all_plumber_scores, "plumber")
    ]

    aggregate_df = pd.DataFrame(aggregate_results)
    print("\n=== Aggregate Comparison Across All Documents ===")
    print(aggregate_df.to_string(index=False))


if __name__ == "__main__":
    main()
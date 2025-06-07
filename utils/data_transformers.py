"""Transform and restructure data for model training and evaluation."""

import pandas as pd
from typing import List, Dict
import logging

logger = logging.getLogger(__name__)


def flatten_extractor_outputs(
    df: pd.DataFrame,
    feature_cols: List[str],
    name_map: Dict[str, str],
    group_col: str = 'gt_sentence_id'
) -> pd.DataFrame:
    """Transform long-format extractor outputs into wide format for model training.
    
    Args:
        df (pd.DataFrame): Input DataFrame in long format.
        feature_cols (List[str]): List of feature columns to pivot.
        name_map (Dict[str, str]): Mapping of extractor names to internal IDs.
        group_col (str, optional): Column to group by. Defaults to 'gt_sentence_id'.
    
    Returns:
        pd.DataFrame: Wide-format DataFrame with features and sentences from all extractors.
    
    Note:
        Creates duplicate rows when multiple extractors tie for highest similarity.
        Always includes 'similarity_score' in pivoting even if not in feature_cols.
        Handles both 'extracted_sentence' and 'matched_extracted_sentence' columns.
        
    Raises:
        ValueError: If required sentence column is missing.
        KeyError: If specified feature columns are not found.
    """
    # Map extractors to internal column-safe names
    df['extractor_id'] = df['extractor'].map(name_map)
    id_to_name = {v: k for k, v in name_map.items()}

    if df['extractor_id'].isnull().any():
        missing = df[df['extractor_id'].isnull()]['extractor'].unique().tolist()
        logger.warning(f"Unmapped extractor names found: {missing}")

    # Always include 'similarity_score' in pivoting, even if not in feature_cols
    full_feature_cols = feature_cols.copy()
    if 'similarity_score' not in full_feature_cols:
        full_feature_cols.append('similarity_score')

    # Pivot sentence-level features
    try:
        pivoted = df.pivot(index=group_col, columns='extractor_id', values=full_feature_cols)
        pivoted.columns = [f"{feat}_{id_to_name[ext_id]}" for feat, ext_id in pivoted.columns]
        pivoted.reset_index(inplace=True)
    except KeyError as e:
        logger.error(f"One or more feature columns missing: {e}")
        raise

    # Pivot extracted sentences (text)
    sentence_col = "extracted_sentence" if "extracted_sentence" in df.columns else "matched_extracted_sentence"
    if sentence_col not in df.columns:
        raise ValueError(f"Expected column '{sentence_col}' not found.")

    sentence_pivot = df.pivot(index=group_col, columns='extractor_id', values=sentence_col)
    sentence_pivot.columns = [f"sentence_{id_to_name[ext_id]}" for ext_id in sentence_pivot.columns]
    sentence_pivot.reset_index(inplace=True)

    # Merge features + sentences
    pivoted = pivoted.merge(sentence_pivot, on=group_col)

    # Select top matches (one or more per GT sentence if tied)
    top_matches = df.groupby(group_col).apply(
        lambda group: group[group['similarity_score'] == group['similarity_score'].max()]
    ).reset_index(drop=True)

    # Add is_tie flag
    top_matches["is_tie"] = top_matches.duplicated(subset=group_col, keep=False)

    # Rename extractor → best_extractor
    top_matches = top_matches.rename(columns={"extractor": "best_extractor"})

    # Final join
    final_df = top_matches[[group_col, "best_extractor", "is_tie"]].merge(pivoted, on=group_col)

    return final_df
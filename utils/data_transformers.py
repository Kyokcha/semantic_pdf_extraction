# utils/data_transformers.py

import pandas as pd
import numpy as np
from typing import List, Dict, Optional
import logging

logger = logging.getLogger(__name__)


def flatten_extractor_outputs(
    df: pd.DataFrame,
    feature_cols: List[str],
    name_map: Dict[str, str],
    group_col: str = 'gt_sentence_id',
    resolve_ties_randomly: bool = True,
    random_state: Optional[int] = 42
) -> pd.DataFrame:
    """
    Flatten a long-format dataframe so that each row corresponds to one ground truth sentence,
    with features from each extractor flattened into separate columns, and a label for the best match.

    Parameters:
    - df: DataFrame with matched sentences and their features.
    - feature_cols: List of feature column names to pivot.
    - name_map: Dict mapping extractor names (e.g., 'ocr') to internal IDs (e.g., 'extractor_1').
    - group_col: Column identifying each GT sentence group.
    - resolve_ties_randomly: Whether to break ties randomly when multiple extractors have top scores.
    - random_state: Random seed for reproducibility.

    Returns:
    - Flattened DataFrame with one row per GT sentence.
    """
    rng = np.random.default_rng(random_state)

    # Reverse map: internal ID -> real name
    id_to_name = {v: k for k, v in name_map.items()}

    if 'extractor_x' not in df.columns:
        raise ValueError("Missing 'extractor_x' column from merged dataset.")

    # Add a consistent extractor_id for pivoting
    df['extractor_id'] = df['extractor_x'].map(name_map)

    if df['extractor_id'].isnull().any():
        missing = df[df['extractor_id'].isnull()]['extractor_x'].unique().tolist()
        logger.warning(f"Unmapped extractor names found: {missing}")

    # Ensure uniqueness of GT sentence ID if layout exists
    if 'layout' in df.columns:
        df[group_col] = df[group_col].astype(str) + '__' + df['layout'].astype(str)

    # Pivot feature columns
    try:
        pivoted = df.pivot(index=group_col, columns='extractor_id', values=feature_cols)
        pivoted.columns = [f"{feat}_{id_to_name[ext_id]}" for feat, ext_id in pivoted.columns]
        pivoted.reset_index(inplace=True)
    except KeyError as e:
        logger.error(f"One or more feature columns are missing: {e}")
        raise

    # Pivot sentence text
    if 'matched_extracted_sentence' not in df.columns:
        raise ValueError("Expected column 'matched_extracted_sentence' not found in input DataFrame.")

    sentence_pivot = df.pivot(index=group_col, columns='extractor_id', values='matched_extracted_sentence')
    sentence_pivot.columns = [f"sentence_{id_to_name[ext_id]}" for ext_id in sentence_pivot.columns]
    sentence_pivot.reset_index(inplace=True)

    # Merge sentences into main pivot
    pivoted = pivoted.merge(sentence_pivot, on=group_col)

    # Identify ties in similarity score
    def is_tie_fn(group):
        return group['similarity_score'].duplicated(keep=False).any()

    tie_flags = (
        df.groupby(group_col)
        .apply(is_tie_fn)
        .reset_index(name='is_tie')
    )

    # Resolve the best extractor (winner)
    def resolve_best(group):
        max_score = group['similarity_score'].max()
        top_rows = group[group['similarity_score'] == max_score]
        if len(top_rows) == 1 or not resolve_ties_randomly:
            return top_rows.iloc[0]['extractor_x']
        return rng.choice(top_rows['extractor_x'].values)

    best_by_score = (
        df.groupby(group_col)
        .apply(resolve_best)
        .reset_index(name='best_extractor')
    )

    # Merge all together
    final_df = pivoted.merge(best_by_score, on=group_col)
    final_df = final_df.merge(tie_flags, on=group_col)

    return final_df

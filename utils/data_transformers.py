# utils/data_transformers.py

import pandas as pd
import numpy as np
from typing import List, Dict, Optional

def flatten_extractor_outputs(
    df: pd.DataFrame,
    feature_cols: List[str],
    name_map: Dict[str, str],
    group_col: str = 'gt_sentence_id',
    resolve_ties_randomly: bool = True,
    random_state: Optional[int] = 42
) -> pd.DataFrame:
    """
    Pivot the dataframe so that each row corresponds to one ground truth sentence,
    with features from each extractor flattened into columns, renamed to actual extractor names.

    Parameters:
    - df: long-format dataframe (one row per extractor output per gt_sentence)
    - feature_cols: list of feature columns to flatten
    - name_map: mapping from real extractor names (e.g. pypdf2) to internal IDs (e.g. extractor_1)
    - group_col: column identifying each ground truth sentence
    - resolve_ties_randomly: if True, randomly pick best extractor when scores are tied
    - random_state: seed for reproducibility if resolving ties randomly

    Returns:
    - Flattened dataframe with one row per ground truth sentence,
      a 'best_extractor' label, and a 'is_tie' flag for score ties
    """
    rng = np.random.default_rng(random_state)

    # Reverse the map to go from internal ID -> real name
    id_to_name = {v: k for k, v in name_map.items()}

    # Map real extractor names to internal IDs for pivoting
    df['extractor_id'] = df['extractor_x'].map(name_map)

    # Update gt_sentence_id to include layout for uniqueness across layout variants
    if 'layout' in df.columns:
        df[group_col] = df[group_col].astype(str) + '__' + df['layout'].astype(str)

    # Pivot the table to have extractor-specific columns
    pivoted = df.pivot(index=group_col, columns='extractor_id', values=feature_cols)

    # Flatten column multi-index and rename using real extractor names
    pivoted.columns = [f"{feat}_{id_to_name[ext_id]}" for feat, ext_id in pivoted.columns]
    pivoted.reset_index(inplace=True)

    # Identify ties: if more than one extractor has the same max similarity score
    def is_tie_fn(group):
        return group['similarity_score'].duplicated(keep=False).any()

    tie_flags = (
        df.groupby(group_col)
        .apply(is_tie_fn)
        .reset_index(name='is_tie')
    )

    # Resolve best extractor
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

    # Merge labels and tie flags into pivoted dataset
    final_df = pivoted.merge(best_by_score, on=group_col)
    final_df = final_df.merge(tie_flags, on=group_col)

    return final_df

"""Main pipeline orchestration script for running the full processing sequence."""

import logging
from utils.config import load_config
import subprocess

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

SCRIPT_SEQUENCE = [
    ("parse_to_json", "scripts/DB_parse_txt_to_json.py"),
    ("run_extractors", "scripts/run_extractors_batch.py"),
    ("generate_gt_sentences", "scripts/generate_gt_sentences.py"),
    ("generate_extracted_sentences", "scripts/generate_extracted_sentences.py"),
    ("extract_features", "scripts/extract_features.py"),
    ("embed_gt", "scripts/embed_groundtruth_sentences.py"),
    ("embed_extracted", "scripts/embed_extracted_sentences.py"),
    ("match_sentences", "scripts/match_by_hungarian_algorithm.py"),
    ("extract_features", "scripts/extract_features.py"),
    ("merged_data", "scripts/merge_matches_with_features.py"),
    ("final_training_dataset", "scripts/build_final_training_dataset.py"),
    ("run_model", "scripts/run_model.py"),
    ("evaluate", "scripts/evaluate_model_semantic_score.py")
]


def run_pipeline() -> None:
    """Execute the full processing pipeline based on configuration.
    
    Runs a sequence of Python scripts in order, with each step controlled
    by the pipeline configuration. Steps can be enabled/disabled via config.
    
    Note:
        Pipeline execution stops if any step returns a non-zero exit code.
        Scripts are run using Poetry to ensure correct environment.
    """
    config = load_config()
    pipeline_config = config.get("pipeline", {})

    for step_key, script_path in SCRIPT_SEQUENCE:
        if pipeline_config.get(step_key, False):
            logger.info(f"➡️ Running step: {step_key}")
            result = subprocess.run(["poetry", "run", "python", "-m", script_path.replace("/", ".").replace(".py", "")])
            if result.returncode != 0:
                logger.error(f"Step failed: {step_key}")
                break
        else:
            logger.info(f"⏭️ Skipping step: {step_key}")


if __name__ == "__main__":
    run_pipeline()
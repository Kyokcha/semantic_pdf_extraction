# scripts/main.py

import logging
from utils.config import load_config
import subprocess

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

SCRIPT_SEQUENCE = [
    ("extract_raw", "scripts/select_sample_txt.py"),
    ("convert_json", "scripts/process_raw_to_JSON.py"),
    ("render_pdfs", "scripts/generate_pdfs.py"),
    ("run_extractors", "scripts/run_extractors_batch.py"),
    ("generate_gt_sentences", "scripts/json_to_DF.py"),
    ("generate_extracted_sentences", "scripts/generate_extracted_sentence_dfs.py"),
    ("embed_gt", "scripts/embed_groundtruth_sentences.py"),
    ("embed_extracted", "scripts/embed_extracted_sentences.py"),
    ("generate_training", "scripts/generate_ranking_training_data.py")
]


def run_pipeline():
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

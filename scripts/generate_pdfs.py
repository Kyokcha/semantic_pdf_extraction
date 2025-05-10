# scripts/generate_pdfs.py

import json
import logging
from pathlib import Path
from multiprocessing import Pool, cpu_count
from utils.config import load_config
from utils.file_operations import clear_directory

# Layout render functions
from layouts import layout_one_column, layout_two_column, layout_header_footer

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Map layout names to their corresponding render functions
LAYOUTS = {
    "one_column": layout_one_column.render,
    "two_column": layout_two_column.render,
    "header_footer": layout_header_footer.render
}


def get_enabled_layouts(config):
    """Return a dictionary of enabled layouts from config."""
    layout_flags = config.get("pdf_generation", {}).get("layouts", {})
    return {name: func for name, func in LAYOUTS.items() if layout_flags.get(name, False)}


def process_file(args):
    """Render all enabled layouts for one JSON file."""
    json_path, pdf_output_dir, enabled_layouts = args

    with open(json_path, "r", encoding="utf-8") as f:
        json_data = json.load(f)

    base_name = json_path.stem
    for layout_name, render_func in enabled_layouts.items():
        output_path = pdf_output_dir / f"{base_name}-{layout_name}.pdf"
        try:
            render_func(json_data, output_path)
            logger.info(f"Generated {output_path.name}")
        except Exception as e:
            logger.error(f"Failed to render {base_name} with layout {layout_name}: {e}")


def main():
    config = load_config()
    pdf_output_dir = Path(config["data_paths"]["pdfs"])
    pdf_output_dir.mkdir(parents=True, exist_ok=True)
    
    # Clear output directory
    clear_directory(pdf_output_dir)

    enabled_layouts = get_enabled_layouts(config)

    if not enabled_layouts:
        logger.warning("No PDF layouts enabled in config. Nothing to do.")
        return

    json_dir = Path(config["data_paths"]["jsons"])
    json_paths = list(json_dir.glob("*.json"))

    logger.info(f"Generating PDFs for {len(json_paths)} JSON files using layouts: {', '.join(enabled_layouts.keys())}")

    # Dynamically select safe number of cores
    available_cores = cpu_count()
    usable_cores = min(8, available_cores - 1)
    logger.info(f"Using {usable_cores} of {available_cores} available CPU cores.")

    # Create a list of arguments to pass to worker processes
    args_list = [(path, pdf_output_dir, enabled_layouts) for path in json_paths]

    with Pool(processes=usable_cores) as pool:
        pool.map(process_file, args_list)

if __name__ == "__main__":
    main()

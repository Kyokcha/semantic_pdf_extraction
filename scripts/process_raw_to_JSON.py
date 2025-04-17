import json
from pathlib import Path
from utils.config import load_config


def main():
# ==== CONFIGURATION ====
    config = load_config()
    raw_dir = Path(config["data_paths"]["raw"])
    ground_truth_dir = Path(config["data_paths"]["ground_truth"])

    # Make output directory for files
    ground_truth_dir.mkdir(parents=True, exist_ok=True)

    # ==== SECTION HEADER HEURISTICS ====
    def is_likely_section_header(line):
        return (
            len(line) <= 80 and
            #line[0].isupper() and
            not line.endswith((".", ":", "!", "?"))
        )

    # ==== MAIN PARSER ====
    def parse_article(file_path):
        with open(file_path, "r", encoding="utf-8") as f:
            lines = [line.rstrip() for line in f]

        # Remove empty lines
        lines = [line for line in lines if line.strip()]

        # First line is assumed to be the title
        title = lines[0].lstrip("# ").strip()
        sections = []
        current_section = {"heading": "Introduction", "paragraphs": []}

        i = 1
        while i < len(lines):
            line = lines[i].strip()

            if is_likely_section_header(line):
                sections.append(current_section)
                current_section = {"heading": line, "paragraphs": []}
                i += 1
            else:
                current_section["paragraphs"].append(line)
                i += 1

        sections.append(current_section)

        return {"title": title, "sections": sections}

    # ==== PROCESS FILES ====
    for file_path in raw_dir.glob("*.txt"):
        parsed = parse_article(file_path)

        out_filename = file_path.stem + ".json"
        out_path = ground_truth_dir / out_filename

        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(parsed, f, indent=2)

    print(f" Finished. JSON ground truth saved to: {ground_truth_dir}")


if __name__ == "__main__":
    main()
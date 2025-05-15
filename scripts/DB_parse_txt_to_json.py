import json
from pathlib import Path
from utils.config import load_config
from utils.file_operations import clear_directory


def parse_txt_file(filepath):
    output = []
    in_table = False

    with open(filepath, 'r', encoding='utf-8') as file:
        for line in file:
            sentence = line.strip()
            if not sentence:
                continue

            if sentence == '[TABLE_START]':
                in_table = True
                continue
            elif sentence == '[TABLE_END]':
                in_table = False
                continue

            if in_table:
                output.append({"type": "table_row", "sentence": sentence})
            elif sentence.startswith('- '):
                output.append({"type": "list_item", "sentence": sentence[2:].strip()})
            else:
                output.append({"type": "text", "sentence": sentence})

    return output


def process_all_txt_files():
    config = load_config()
    input_dir = Path(config["data_paths"]["DB_raw_manual"])
    output_dir = Path(config["data_paths"]["DB_jsons"])
    output_dir.mkdir(parents=True, exist_ok=True)
    clear_directory(output_dir)

    for txt_file in input_dir.glob("*.txt"):
        parsed_data = parse_txt_file(txt_file)
        output_file = output_dir / (txt_file.stem + ".json")
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(parsed_data, f, indent=2, ensure_ascii=False)
        print(f"Parsed: {txt_file.name} -> {output_file.name}")


if __name__ == "__main__":
    process_all_txt_files()

# utils/section_grouper.py

from typing import List, Dict
from collections import defaultdict
import statistics
import logging


def parse_file(filepath: str) -> List[Dict]:
    """
    Parses a tab-separated .txt file into tokens with spatial info.
    Assumes format: word x0 y0 x1 y1 ... font block_type
    """
    tokens = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) < 10:
                continue

            try:
                word, x0, y0, x1, y1 = parts[0], int(parts[1]), int(parts[2]), int(parts[3]), int(parts[4])
                font = parts[-2]
                block_type = parts[-1]
                tokens.append({
                    "word": word,
                    "x0": x0,
                    "y0": y0,
                    "x1": x1,
                    "y1": y1,
                    "font": font,
                    "type": block_type,
                })
            except ValueError:
                continue
    return tokens


def detect_columns_in_band(tokens: List[Dict], band_height=100, gap_threshold=100) -> List[List[Dict]]:
    """
    Detects columns dynamically in vertical bands.
    Splits each band into column groups if a large X-gap is detected.
    """
    bands = defaultdict(list)
    for token in tokens:
        band_idx = token["y0"] // band_height
        bands[band_idx].append(token)

    column_bands = []
    for _, band_tokens in sorted(bands.items()):
        if not band_tokens:
            continue

        band_tokens.sort(key=lambda t: t["x0"])
        x_values = [t["x0"] for t in band_tokens]
        if max(x_values) - min(x_values) > gap_threshold:
            midpoint = (max(x_values) + min(x_values)) / 2
            left = [t for t in band_tokens if t["x0"] <= midpoint]
            right = [t for t in band_tokens if t["x0"] > midpoint]
            column_bands.append(left)
            column_bands.append(right)
        else:
            column_bands.append(band_tokens)

    return column_bands


def group_by_lines(tokens: List[Dict], y_tolerance=5) -> List[List[Dict]]:
    """
    Groups tokens into lines based on vertical (y-axis) overlap.
    """
    sorted_tokens = sorted(tokens, key=lambda t: (t["y0"], t["x0"]))
    lines = []

    for token in sorted_tokens:
        matched = False
        for line in lines:
            if abs(token["y0"] - line[0]["y0"]) <= y_tolerance:
                line.append(token)
                matched = True
                break
        if not matched:
            lines.append([token])

    return lines


def group_lines_by_type(lines: List[List[Dict]]) -> List[Dict]:
    """
    Groups lines into sections based on block_type continuity.
    """
    sections = []
    current_section = {"type": None, "text": "", "bbox": [9999, 9999, -1, -1]}

    def update_bbox(bbox, token):
        bbox[0] = min(bbox[0], token["x0"])
        bbox[1] = min(bbox[1], token["y0"])
        bbox[2] = max(bbox[2], token["x1"])
        bbox[3] = max(bbox[3], token["y1"])
        return bbox

    for line in lines:
        line = sorted(line, key=lambda t: t["x0"])
        text = " ".join([t["word"] for t in line])
        block_type = line[0]["type"]

        if current_section["type"] != block_type and current_section["text"]:
            sections.append(current_section)
            current_section = {"type": block_type, "text": "", "bbox": [9999, 9999, -1, -1]}

        current_section["type"] = block_type
        current_section["text"] += (" " + text).strip()
        for token in line:
            current_section["bbox"] = update_bbox(current_section["bbox"], token)

    if current_section["text"]:
        sections.append(current_section)

    return sections


def adaptive_reading_order(sections: List[Dict], band_height=100) -> List[Dict]:
    """
    Determines reading order by analyzing horizontal gaps between tokens.
    Sorts sections top-down or left-then-down per vertical band.
    """
    if not sections:
        return []

    bands = defaultdict(list)
    for sec in sections:
        band_idx = sec["bbox"][1] // band_height
        bands[band_idx].append(sec)

    ordered_sections = []

    for _, band_sections in sorted(bands.items()):
        tokens = []
        for sec in band_sections:
            tokens.append({"x0": sec["bbox"][0], "x1": sec["bbox"][2]})

        tokens = sorted(tokens, key=lambda t: t["x0"])
        gaps = [tokens[i]["x0"] - tokens[i - 1]["x1"] for i in range(1, len(tokens)) if tokens[i]["x0"] > tokens[i - 1]["x1"]]

        if len(gaps) < 3:
            band_sections.sort(key=lambda s: s["bbox"][1])  # default top-down
        else:
            avg = statistics.mean(gaps)
            stddev = statistics.stdev(gaps)
            large_gaps = [g for g in gaps if g > avg + stddev]
            is_multicol = len(large_gaps) > len(gaps) / 4

            if is_multicol:
                band_sections.sort(key=lambda s: (s["bbox"][0], s["bbox"][1]))
            else:
                band_sections.sort(key=lambda s: s["bbox"][1])

        ordered_sections.extend(band_sections)

    return ordered_sections


def extract_sections(filepath: str) -> List[Dict]:
    """
    Full extraction pipeline:
    - parse file into tokens
    - detect column bands
    - group into lines and sections
    - apply adaptive reading order
    """
    tokens = parse_file(filepath)
    logging.debug(f"Parsed {len(tokens)} tokens from {filepath}")

    column_bands = detect_columns_in_band(tokens)
    all_sections = []

    for band_tokens in column_bands:
        lines = group_by_lines(band_tokens)
        sections = group_lines_by_type(lines)
        all_sections.extend(sections)

    return adaptive_reading_order(all_sections)

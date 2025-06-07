# PDF Text Extraction Pipeline

A comprehensive pipeline for extracting and evaluating text from PDF documents using multiple extraction methods.

## Overview

This project implements a machine learning pipeline that:
1. Extracts text from PDFs using multiple methods (OCR, PyPDF2, PDFPlumber)
2. Evaluates and compares the quality of different extraction methods
3. Trains a model to select the best extraction method for each piece of text

## Features

- Multiple PDF text extraction methods:
  - OCR using Tesseract
  - Direct extraction using PyPDF2
  - Layout-aware extraction using PDFPlumber
- Sentence-level comparison and matching
- Feature generation for extraction quality assessment
- Model training for automatic extractor selection
- Parallel processing for efficiency

## Project Structure

```
├── configs/          # Configuration files
├── extractors/       # PDF text extraction modules
│   ├── extract_ocr.py
│   ├── extract_pdfplumber.py
│   └── extract_pypdf2.py
├── models/          # ML model implementations
│   ├── base.py
│   ├── lightgbm_model.py
│   ├── random_forest.py
│   └── xgboost_model.py
├── scripts/         # Processing pipeline scripts
│   ├── run_extractors_batch.py
│   ├── generate_extracted_sentences.py
│   └── ...
└── utils/          # Utility functions
    ├── embedding.py
    ├── pdf_utils.py
    └── ...
```

## Installation

1. Clone the repository
2. Install dependencies:
   ```bash
   poetry install
   ```
3. Install required model files:
   ```bash
   python -m scripts.setup_environment
   ```

## Usage

1. Configure the pipeline in `config.yaml`
2. Run the complete pipeline:
   ```bash
   python -m scripts.main
   ```

Or run individual steps:
```bash
# Process DocBank dataset
python -m scripts.rename_and_organise_docbank

# Run extractors
python -m scripts.run_extractors_batch

# Generate features and train model
python -m scripts.run_model
```

## Pipeline Steps

1. **Data Preparation**
   - Organize and preprocess PDF files
   - Generate ground truth data

2. **Text Extraction**
   - Run multiple extractors in parallel
   - Generate sentence-level extractions

3. **Feature Generation**
   - Compute linguistic features
   - Generate embeddings
   - Compare extractor outputs

4. **Model Training**
   - Train classifier to select best extractor
   - Evaluate performance
   - Generate performance reports

## Configuration

Key configuration options in `config.yaml`:
- Data paths
- Enabled extractors
- Feature selection
- Model parameters

## Models

Supported classifiers:
- Random Forest
- LightGBM
- XGBoost

## Requirements

- Python 3.8+
- Tesseract OCR
- Poetry for dependency management

## License

[Add license information]
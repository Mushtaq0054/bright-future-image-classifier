# Image Classification for Bright Future School: Classroom vs Playground

Automated image classification pipeline developed for Bright Future School.

## Task 1 Deliverables
- Modular directory architecture (`src/`, `data/`, `docs/`).
- Automated data download and 3-way split pipeline script (`train/`, `val/`, `test/`).
- Architecture Decision Log (ADL) documenting design decisions and splitting strategy.

## Project Structure
```text
bright-future-image-classifier/
├── data/
│   ├── raw/
│   ├── train/
│   ├── val/
│   └── test/
├── docs/
│   └── architecture_decision_log.md
├── src/
│   ├── __init__.py
│   └── data_pipeline.py
├── .gitignore
├── requirements.txt
└── README.md
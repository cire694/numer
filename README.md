# Numerai Data Science Pipeline

A modular, production-ready pipeline for the Numerai data science contest. The architecture separates concerns into distinct modules: only you write the model, everything else is fixed and reusable.

## Project Structure

```
numer/
├── __init__.py              # Package initialization
├── config.py                # Configuration management (dataclass)
├── data.py                  # Data loading & caching
├── models.py                # Model registry & factory
├── pipeline.py              # Training pipeline orchestration
├── evaluate.py              # Evaluation metrics
├── submission.py            # Submission handling
├── utils.py                 # Model persistence utilities
├── main.py                  # CLI entry point
├── pyrightconfig.json       # Type checking configuration
└── README.md                # This file
```

## Quick Start

### Installation

```bash
pip install numerapi xgboost scikit-learn pandas
```

### Train & Submit

```bash
# Train and immediately submit predictions
python main.py --mode train-submit --model ridge --data-version v5.2

# Only train (save model)
python main.py --mode train --model xgboost

# Only submit with latest model
python main.py --mode submit --model ridge
```







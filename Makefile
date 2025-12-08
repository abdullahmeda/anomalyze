.PHONY: env install dataset prepare visualize prophet arima test clean help

# Default target
help:
	@echo "Available commands:"
	@echo "  make env         - Create a local virtual environment in .venv and upgrade pip"
	@echo "  make install     - Install package with dev dependencies"
	@echo "  make dataset     - Generate full dataset from HuggingFace"
	@echo "  make prepare     - Create train/test splits for ML training"
	@echo "  make visualize   - Generate time-series plots"
	@echo "  make prophet     - Run Prophet anomaly detection"
	@echo "  make arima       - Run ARIMA anomaly detection"
	@echo "  make test        - Run all tests"
	@echo "  make clean       - Remove generated files"


# Create a local virtual environment in .venv and upgrade pip
env:
	python3 -m venv .venv
	. .venv/bin/activate && pip install --upgrade pip

# Install package in editable mode with dev dependencies
install:
	. .venv/bin/activate && pip install -e ".[dev]"

# Generate the raw dataset from HuggingFace
dataset:
	. .venv/bin/activate && python3 -m dataset.preprocess

# Create train/test splits for ML training
prepare:
	. .venv/bin/activate && python3 -m ml.features

# Generate visualization plots
visualize:
	. .venv/bin/activate && python3 -m dataset.visualize

# Run Prophet anomaly detection
prophet:
	. .venv/bin/activate && python3 -m ml.run --model prophet

# Run ARIMA anomaly detection
arima:
	. .venv/bin/activate && python3 -m ml.run --model arima

# Run tests
test:
	. .venv/bin/activate && python3 -m pytest dataset/tests.py -v

# Clean generated files
clean:
	rm -rf dataset/data/*.csv
	rm -rf dataset/data/anomaly_metadata.json
	rm -rf dataset/plots/*.png
	rm -rf ml/data/*.csv
	rm -rf __pycache__ */__pycache__ */*/__pycache__
	rm -rf .pytest_cache */.pytest_cache
	rm -rf *.egg-info


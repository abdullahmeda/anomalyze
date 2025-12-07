.PHONY: install prepare visualize test clean help

# Default target
help:
	@echo "Available commands:"
	@echo "  make install     - Install package with dev dependencies"
	@echo "  make prepare     - Generate dataset from HuggingFace"
	@echo "  make visualize   - Generate time-series plots"
	@echo "  make test        - Run all tests"
	@echo "  make clean       - Remove generated files"

# Install package in editable mode with dev dependencies
install:
	pip install -e ".[dev]"

# Generate the dataset
prepare:
	cd dataset && python prepare.py

# Generate visualization plots
visualize:
	cd dataset && python visualize.py

# Run tests
test:
	cd dataset && pytest tests.py -v

# Clean generated files
clean:
	rm -rf dataset/data/*.csv
	rm -rf dataset/data/anomaly_metadata.json
	rm -rf dataset/plots/*.png
	rm -rf __pycache__ */__pycache__ */*/__pycache__
	rm -rf .pytest_cache */.pytest_cache
	rm -rf *.egg-info


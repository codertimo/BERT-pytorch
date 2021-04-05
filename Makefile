.PHONY: style quality test test-cov

check_dirs := bert/ scripts/ tests/
test_dirs := bert/

style:
	black $(check_dirs)
	isort $(check_dirs)
	flake8 $(check_dirs)

quality:
	black --check $(check_dirs)
	isort --check-only $(check_dirs)
	flake8 $(check_dirs)

test:
	pytest

test-cov:
	pytest --cov-branch --cov $(test_dirs)

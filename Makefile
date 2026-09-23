.PHONY: install train eda test

install:
	pip install -r requirements.txt

train:
	python scripts/train.py

eda:
	python scripts/eda.py

test:
	python -m unittest discover -s tests -v

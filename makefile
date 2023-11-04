SHELL := /usr/bin/bash
make:
	make .venv
	make requirements.txt

.venv:
	python3 -m venv .venv

.PHONY: requirements.txt
requirements.txt: .venv
	source .venv/bin/activate; \
	pip freeze > requirements.txt

.PHONY: clean
clean:
	rm -rf .venv requirements.txt
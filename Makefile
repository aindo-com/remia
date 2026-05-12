.ONESHELL:
SHELL := /bin/bash

PYTHON_VERSION ?= 3.12

# Installation
.venv:
	( \
		python$(PYTHON_VERSION) -m venv .venv && \
		source .venv/bin/activate && \
		python -m pip install -U pip setuptools cython \
	) || rm -rf .venv

.PHONY: install
install: .venv
	source .venv/bin/activate && \
	python -m pip install -r requirements.txt

# Extracting experiments
experiments:
	tar -xJf experiments.tar.xz
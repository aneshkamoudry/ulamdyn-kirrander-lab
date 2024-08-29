SHELL := /bin/bash

ENV_NAME = venv
ENV_INSTALL_NAME = install
PYTHON = python3

# get package name and version from setup file
PACKAGE_NAME :=  $(shell python3 setup.py --name)
PACKAGE_VERSION := $(shell python3 setup.py --version)

# name convention for the generated wheel file
WHEEL_NAME := dist/$(PACKAGE_NAME)-$(PACKAGE_VERSION)-py3-none-any.whl

# list of python files used to build the wheel file
PY_FILES = $(shell find src/ -type f -name '*.py')

# list of folders/files considered as source code (used by linters)
SOURCE_CODE = src setup.py

# clean environment
clean:
	@echo ">>> clean env:" $(ENV_NAME)
	rm -rf ./$(ENV_NAME)
	rm -rf __pycache__
	rm -rf ./build
	rm -rf ./dist
	rm -rf ./*.egg-info
	rm -rf ./src/*.egg-info
	rm -rf .coverage
	rm -rf coverage.xml
	rm -rf coverage_html

# create virtual environment with all dependencies
venv: requirements.txt
	@echo Creating Python virtual env : $(ENV_NAME)
	${PYTHON} -m venv $(ENV_NAME)
	source ./$(ENV_NAME)/bin/activate
	@echo Upgrading pip version
	./$(ENV_NAME)/bin/python3 -m pip install --upgrade pip
	@echo Installing wheel package and upgrade setuptools
	./$(ENV_NAME)/bin/python3 -m pip install wheel
	./$(ENV_NAME)/bin/python3 -m pip install --upgrade setuptools
	@echo Installing dependencies from requirements.txt
	./$(ENV_NAME)/bin/python3 -m pip install -r requirements.txt

# run linter checks (and pytests whenever available) 
check:
	@echo Running pylint in the source code...
	${ENV_NAME}/bin/pylint ${SOURCE_CODE}
	@echo Running isort...
	${ENV_NAME}/bin/isort --profile black --check ${SOURCE_CODE}
	@echo Checking code convention with black...
	${ENV_NAME}/bin/black --check ${SOURCE_CODE}
#	@echo Running tests...
#	${ENV_NAME}/bin/pytest -s --cov=src tests --cov-report term --cov-report html:coverage_html --cov-report xml:coverage.xml

# Build Python package
# create an alias "build" to generate wheel file if python files has changed
make build: $(WHEEL_NAME)
# generate wheel file if python files has changed
$(WHEEL_NAME): ${ENV_NAME} $(PY_FILES)
	${ENV_NAME}/bin/python setup.py bdist_wheel

# Install package
# alias for ${VENV_INSTALL_NAME}
install: ${ENV_INSTALL_NAME}

.PHONY: ${ENV_INSTALL_NAME}
${ENV_INSTALL_NAME}: build
	rm -rf ${ENV_INSTALL_NAME}
	${PYTHON} -m venv ${ENV_INSTALL_NAME}
	${ENV_INSTALL_NAME}/bin/pip install -r requirements.txt
	${ENV_INSTALL_NAME}/bin/pip install $(WHEEL_NAME)

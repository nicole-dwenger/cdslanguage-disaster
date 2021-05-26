#!/usr/bin/env bash

VENVNAME=venv_disaster

python3 -m venv $VENVNAME
source $VENVNAME/bin/activate
pip install --upgrade pip

# requirements
test -f requirements.txt && pip install -r requirements.txt

deactivate
echo "build $VENVNAME"

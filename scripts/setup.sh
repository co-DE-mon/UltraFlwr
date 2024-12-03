#!/usr/bin/env bash

# Check if the user provided a virtual environment name
if [ -z "$1" ]; then
    echo "Usage: $0 <venv_name>"
    exit 1
fi

VENV_NAME=$1

# navigate to directory
SCRIPTPATH="$( cd "$(dirname "$0")" ; pwd -P )"
cd $SCRIPTPATH
cd ..

python3 -m venv "$VENV_NAME"

source "$VENV_NAME/bin/activate"

pip install --upgrade pip

pip install -e .

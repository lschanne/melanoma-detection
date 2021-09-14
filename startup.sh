#!/usr/bin/bash
echo "starting up the melanoma detection project..."

source venv_melanoma_detection/bin/activate

export FLASK_ENV=development
export FLASK_APP=melanoma_detection

python -m flask run --host=0.0.0.0

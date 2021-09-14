#!/usr/bin/bash
echo "installing the melanoma detection project..."
python3 -m venv venv_melanoma_detection
source venv_melanoma_detection/bin/activate
python3 -m pip install -r requirements.txt

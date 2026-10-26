#!/bin/bash
# Local upload-and-download front end for the pipeline.
set -e
cd "$(dirname "$0")"
export OLLAMA_KEEP_ALIVE=0
echo "open http://127.0.0.1:5001"
exec ./.venv/bin/python src/webapp.py

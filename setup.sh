#!/bin/bash
# Setup script for Streamlit Cloud
# Runs BEFORE requirements.txt to prepare the environment
set -e

echo "=== Streamlit Cloud setup ==="
echo "Python version: $(python --version)"
echo "Pip version: $(pip --version)"

# Pin pip to a known-working version to avoid auto-upgrade issues
echo "=== Pinning pip version ==="
pip install --quiet "pip==24.0" || echo "pip pin skipped"

# Pre-install build dependencies that scikit-learn needs
echo "=== Pre-installing Cython and NumPy ==="
pip install --quiet "Cython>=3.0.0,<4.0.0" || echo "Cython install skipped"

echo "=== Setup complete ==="

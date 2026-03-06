#!/bin/bash
# Colab Setup Script for Face-Swap API
# Run this ONCE after cloning, then restart the Colab runtime.

set -e

echo "=== Installing numpy 1.26.4 (must be 1.x for onnxruntime) ==="
pip install numpy==1.26.4

echo "=== Installing onnxruntime-gpu ==="
pip install onnxruntime-gpu==1.18.0

echo "=== Installing project requirements ==="
pip install -r requirements.txt

echo "=== Upgrading opennsfw2 and keras ==="
pip install opennsfw2 keras --upgrade

echo "=== Removing jax/jaxlib (conflict prevention) ==="
pip uninstall -y jax jaxlib 2>/dev/null || true

echo "=== Force-pinning numpy back to 1.26.4 (other deps may have upgraded it) ==="
pip install numpy==1.26.4 --force-reinstall

echo ""
echo "============================================"
echo "  SETUP COMPLETE"
echo "  NOW RESTART THE COLAB RUNTIME:"
echo "  Runtime -> Restart runtime"
echo "============================================"

#!/bin/bash
# Colab Setup Script for Face-Swap API
# Run: !bash setup_colab.sh
# Then: Restart the Colab runtime (Runtime -> Restart runtime)

set -e

echo "=== Step 1: Installing onnxruntime-gpu (numpy 2.x compatible) ==="
pip install onnxruntime-gpu>=1.20.0

echo "=== Step 2: Installing project requirements ==="
pip install -r requirements.txt

echo "=== Step 3: Upgrading opennsfw2 and keras ==="
pip install opennsfw2 keras --upgrade

echo "=== Step 4: Removing jax/jaxlib (conflict prevention) ==="
pip uninstall -y jax jaxlib 2>/dev/null || true

echo "=== Step 5: Fixing basicsr + torchvision compatibility ==="
# basicsr uses a deprecated import that was removed in newer torchvision
DEGRADATIONS_FILE=$(python3 -c "import basicsr; import os; print(os.path.join(os.path.dirname(basicsr.__file__), 'data', 'degradations.py'))")
if [ -f "$DEGRADATIONS_FILE" ]; then
    sed -i 's/from torchvision.transforms.functional_tensor import rgb_to_grayscale/from torchvision.transforms.functional import rgb_to_grayscale/' "$DEGRADATIONS_FILE"
    echo "  Patched: $DEGRADATIONS_FILE"
else
    echo "  WARNING: Could not find degradations.py to patch"
fi

echo ""
echo "============================================"
echo "  SETUP COMPLETE!"
echo "  NOW RESTART THE COLAB RUNTIME:"
echo "  Runtime -> Restart runtime"
echo "============================================"

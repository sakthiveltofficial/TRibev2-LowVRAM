#!/usr/bin/env bash
# ==============================================================================
# download_algonauts.sh — Helper script to fetch Algonauts 2025 / OpenNeuro data
# ==============================================================================

set -euo pipefail

DEST_DIR="${1:-data/algonauts2025}"
mkdir -p "$DEST_DIR"

echo "=========================================================="
echo " Fetching Brain Encoding Data for TRIBE-local"
echo " Destination: $DEST_DIR"
echo "=========================================================="

echo "Options for downloading real datasets:"
echo "1. Algonauts 2025 (Courtois NeuroMod movie fMRI dataset):"
echo "   Visit: https://algonautsproject.csail.mit.edu/ or"
echo "   Use DataLad: datalad install https://github.com/courtois-neuromod/cneuromod"
echo ""
echo "2. OpenNeuro Lahner 2024 (BOLD Moments, 3s video clips):"
echo "   pip install openneuro-py"
echo "   openneuro-py download --dataset ds004873 --target-dir $DEST_DIR/lahner2024"
echo ""
echo "3. OpenNeuro Lebel 2023 (Natural Language narratives):"
echo "   openneuro-py download --dataset ds003020 --target-dir $DEST_DIR/lebel2023"
echo ""
echo "4. Or generate immediate synthetic validation data by running:"
echo "   python -c 'from tribe_local.data.preprocessing import create_synthetic_dataset; create_synthetic_dataset(\"data/synthetic\")'"
echo "=========================================================="

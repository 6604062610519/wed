#!/bin/bash
# run_compare.sh — Script สำหรับรัน wildfire model comparison
# Usage:
#   bash run_compare.sh quick    # prototype (5 epochs, ไม่มี tuning)
#   bash run_compare.sh full     # full comparison (30 epochs + tuning)
#   bash run_compare.sh rf       # เฉพาะ Random Forest baseline
#   bash run_compare.sh cnn      # เฉพาะ CNN

set -e
cd "$(dirname "$0")"

MODE=${1:-quick}
mkdir -p /tmp/mpl_cache checkpoints results

export MPLCONFIGDIR=/tmp/mpl_cache

echo "🔥 Wildfire Model Comparison"
echo "   Mode: $MODE"
echo "   Device: Apple Silicon M2 (MPS)"
echo ""

case "$MODE" in
  quick)
    echo "▶️  Running QUICK benchmark (5 epochs, no tuning)..."
    python3 compare.py --mode quick --no-tune
    ;;
  full)
    echo "▶️  Running FULL comparison (30 epochs + HP tuning)..."
    python3 compare.py --mode full
    ;;
  rf)
    echo "▶️  Running Random Forest only..."
    python3 compare.py --mode quick --models rf --no-tune
    ;;
  cnn)
    echo "▶️  Running CNN only..."
    python3 compare.py --mode quick --models cnn --no-tune
    ;;
  unet)
    echo "▶️  Running U-Net only..."
    python3 compare.py --mode quick --models unet --no-tune
    ;;
  resunet)
    echo "▶️  Running ResU-Net only..."
    python3 compare.py --mode quick --models resunet --no-tune
    ;;
  convlstm)
    echo "▶️  Running ConvLSTM only..."
    python3 compare.py --mode quick --models convlstm --no-tune
    ;;
  dl)
    echo "▶️  Running all DL models (no RF)..."
    python3 compare.py --mode quick --models cnn unet resunet convlstm --no-tune
    ;;
  *)
    echo "Usage: bash run_compare.sh [quick|full|rf|cnn|unet|resunet|convlstm|dl]"
    exit 1
    ;;
esac

echo ""
echo "✅ Done! Results in: results/"
echo "📊 Chart: results/comparison_chart.png"

#!/usr/bin/env bash
# Quick start for Linux / macOS.
# Usage:
#   ./setup_and_run.sh install      # CPU torch (default)
#   ./setup_and_run.sh install_gpu  # CUDA 12.1 torch
#   ./setup_and_run.sh train        # train on GPU
#   ./setup_and_run.sh web          # launch Flask app
#   ./setup_and_run.sh all          # install (CPU) + launch web
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"

ACTION="${1:-all}"

install_cpu() {
  python3 -m pip install --upgrade pip
  python3 -m pip install -r "$ROOT/requirements_pytorch.txt" \
    --index-url https://download.pytorch.org/whl/cpu \
    --extra-index-url https://pypi.org/simple
}
install_gpu() {
  python3 -m pip install --upgrade pip
  python3 -m pip install -r "$ROOT/requirements_pytorch.txt" \
    --index-url https://download.pytorch.org/whl/cu121 \
    --extra-index-url https://pypi.org/simple
}
train() {
  cd "$ROOT"
  python3 -m pytorch_impl.train --data "$ROOT/data/lol_dataset" \
    --img-size 512 --epochs 180 --batch-size 1 --critic-updates 5 \
    --save-dir "$ROOT/weights"
}
web() {
  cd "$ROOT"
  EDNIG_HOST="${EDNIG_HOST:-127.0.0.1}" EDNIG_PORT="${EDNIG_PORT:-5000}" \
    python3 "$ROOT/webapp/app.py"
}

case "$ACTION" in
  install)     install_cpu ;;
  install_gpu) install_gpu ;;
  train)       train ;;
  web)         web ;;
  all)         install_cpu; web ;;
  *) echo "Unknown action: $ACTION"; exit 1 ;;
esac

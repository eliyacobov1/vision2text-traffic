#!/bin/bash

# Simple wrapper to launch the Streamlit demo with sensible defaults.
# Users can optionally pass ``--ckpt`` and ``--config`` paths which will be
# forwarded to ``demo.py``. If a path is missing the script falls back to
# defaults and prints a warning instead of failing.

CKPT=""
CONFIG=""
ARGS=()

while [[ "$#" -gt 0 ]]; do
  case $1 in
    --ckpt)
      CKPT="$2"; shift 2;;
    --config)
      CONFIG="$2"; shift 2;;
    *)
      ARGS+=("$1"); shift;;
  esac
done

# Default paths if none provided
CKPT=${CKPT:-checkpoints/model.pt}
CONFIG=${CONFIG:-config.yaml}

if [ ! -f "$CKPT" ]; then
  echo "[run_demo] Warning: checkpoint '$CKPT' not found; the model will use random weights." >&2
fi

CONFIG_ARG=""
if [ -f "$CONFIG" ]; then
  CONFIG_ARG="--config $CONFIG"
else
  echo "[run_demo] Warning: config '$CONFIG' not found; internal defaults will be used." >&2
fi

streamlit run demo.py -- --ckpt "$CKPT" $CONFIG_ARG "${ARGS[@]}"

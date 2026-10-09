#!/usr/bin/env bash
# Start JARVIS with GPU Whisper + the 3B model.
# Put this file next to jarvis.py, then:  chmod +x run_jarvis.sh && ./run_jarvis.sh
#
#   JARVIS_CPU=1 ./run_jarvis.sh       -> force everything on CPU (Whisper base)
#   JARVIS_LLM=llama3.2:1b ./run_jarvis.sh   -> use a different Ollama model

cd "$(dirname "$0")" || exit 1

# 1) find and activate the virtual environment
for cand in "$JARVIS_VENV" "$HOME/JARVIS/venv" "$HOME/JARVIS/.venv" "$HOME/JARVIS"; do
    if [ -n "$cand" ] && [ -f "$cand/bin/activate" ]; then
        # shellcheck disable=SC1090
        source "$cand/bin/activate"
        echo "Using virtual environment: $cand"
        break
    fi
done
if [ -z "$VIRTUAL_ENV" ]; then
    echo "No virtual environment found. Set JARVIS_VENV=/path/to/venv and try again."
    exit 1
fi

# 2) let faster-whisper (CTranslate2) find the CUDA 12 libraries installed by pip
# (nvidia.*.lib are namespace packages: __file__ is None, so use __path__)
NVLIBS=$(python - <<'EOF' 2>/dev/null
try:
    import nvidia.cublas.lib, nvidia.cudnn.lib
    print(nvidia.cublas.lib.__path__[0] + ":" + nvidia.cudnn.lib.__path__[0])
except Exception:
    pass
EOF
)
if [ -n "$NVLIBS" ]; then
    export LD_LIBRARY_PATH="$NVLIBS:$LD_LIBRARY_PATH"
else
    echo "Note: nvidia-cublas-cu12 / nvidia-cudnn-cu12 are not installed in this venv."
    echo "      JARVIS will start, but Whisper will fall back to the CPU."
fi

# 3) default to the 3B model (fits in 4 GB VRAM); override with JARVIS_LLM
export JARVIS_LLM="${JARVIS_LLM:-llama3.2:3b}"
if [ "$JARVIS_CPU" = "1" ]; then export JARVIS_DEVICE=cpu; fi

exec python jarvis.py

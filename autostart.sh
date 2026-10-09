#!/usr/bin/env bash
# Autostart wrapper for JARVIS (launched by ~/.config/autostart/jarvis.desktop).
#
# Why a wrapper instead of putting run_jarvis.sh straight into the .desktop file:
#   1) the desktop session (panel, audio, Ollama) needs a moment to settle before
#      JARVIS grabs the microphone and loads models;
#   2) we wait (bounded) until Ollama answers so the first question isn't a
#      connection error;
#   3) everything is logged to ~/.jarvis/startup.log so a silent failure at login
#      is diagnosable instead of invisible.

LOG="$HOME/.jarvis/startup.log"
mkdir -p "$(dirname "$LOG")"

{
    echo "==============================================================="
    echo "JARVIS autostart  $(date '+%Y-%m-%d %H:%M:%S')"
    echo "DISPLAY=$DISPLAY  XAUTHORITY=${XAUTHORITY:-unset}  USER=$USER"
} >>"$LOG" 2>&1

# 1) let the session settle (Cinnamon panel, pulseaudio, network)
sleep 15

# 2) wait up to ~60s for Ollama to be reachable
for _ in $(seq 1 30); do
    if curl -s -o /dev/null --max-time 2 http://127.0.0.1:11434/api/tags; then
        echo "Ollama: reachable" >>"$LOG" 2>&1
        break
    fi
    sleep 2
done

# 3) start JARVIS (unbuffered so the log stays live even if it crashes)
export PYTHONUNBUFFERED=1
echo "Starting run_jarvis.sh ..." >>"$LOG" 2>&1
exec "$HOME/JARVIS/run_jarvis.sh" >>"$LOG" 2>&1

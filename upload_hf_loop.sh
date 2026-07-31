#!/bin/bash
# Auto-restart upload until "All uploads complete" appears
LOG=/Users/hc/Documents/research/Projects/dataset_RFSS/upload_hf.log
cd /Users/hc/Documents/research/Projects/dataset_RFSS

while true; do
    if grep -q "All uploads complete" "$LOG" 2>/dev/null; then
        echo "[wrapper] Upload complete. Exiting." >> "$LOG"
        break
    fi
    echo "[wrapper] Starting upload at $(date)" >> "$LOG"
    uv run python3 -u upload_hf.py >> "$LOG" 2>&1
    EXIT=$?
    if grep -q "All uploads complete" "$LOG" 2>/dev/null; then
        echo "[wrapper] Upload complete. Exiting." >> "$LOG"
        break
    fi
    echo "[wrapper] Process exited (code $EXIT). Restarting in 10s..." >> "$LOG"
    sleep 10
done

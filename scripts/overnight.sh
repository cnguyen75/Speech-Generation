#!/bin/zsh
# Overnight launcher: keeps the Mac awake for exactly as long as the study runs, works offline
# from the cached models, and resumes up to 3 times after a crash (finished runs are skipped).
cd "${0:A:h}/.."
mkdir -p runs
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
for attempt in 1 2 3 4; do
  caffeinate -ims .venv/bin/python -u scripts/run_study.py >> runs/stdout.log 2>&1
  code=$?
  [[ $code == 0 ]] && exit 0
  [[ $code == 2 ]] && { echo "$(date '+%F %T') stopped: repeated run failures (exit 2)" >> runs/study.log; exit 2; }
  echo "$(date '+%F %T') job exited with code $code (attempt $attempt); resuming in 30 s" >> runs/study.log
  sleep 30
done

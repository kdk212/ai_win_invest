#!/usr/bin/env bash
set -euo pipefail

APP_DIR="/home/ubuntu/ai_win_invest"
START_DATE="${START_DATE:-2026-06-01}"
END_DATE="${END_DATE:-$(date +%F)}"
WINDOWS="${WINDOWS:-12,18,24}"
LOG_DIR="$APP_DIR/logs"
LOG_FILE="$LOG_DIR/rebuild_from_2026-06-01.log"

cd "$APP_DIR"
mkdir -p "$LOG_DIR"
{
  echo "Rebuild started: $(date -Iseconds)"
  echo "Start date: $START_DATE"
  echo "End date: $END_DATE"
  echo "Windows: $WINDOWS"
  echo ""

  .venv/bin/python -m py_compile \
    ai_invest/strategy.py \
    ai_invest/optimizer.py \
    ai_invest/monitor.py \
    ai_invest/virtual_portfolio.py \
    ai_invest/cli.py

  .venv/bin/python main.py optimize-strategy --windows "$WINDOWS"

  find data/recommendations -name "recommendations_2026-06-*.csv" -delete
  .venv/bin/python main.py backfill-recommendations --start "$START_DATE" --end "$END_DATE" --with-news
  .venv/bin/python main.py monitor-strategy --auto-optimize
  .venv/bin/python main.py strategy-status > "$LOG_DIR/strategy_status.txt"

  sudo systemctl restart ai-invest
  echo ""
  echo "Rebuild finished: $(date -Iseconds)"
} > "$LOG_FILE" 2>&1

cat "$LOG_DIR/strategy_status.txt"

#!/usr/bin/env bash
set -euo pipefail

APP_DIR="/home/ubuntu/ai_win_invest"
START_DATE="${START_DATE:-2026-06-01}"
END_DATE="${END_DATE:-$(date +%F)}"
WINDOWS="${WINDOWS:-12,18,24}"
WITH_NEWS="${WITH_NEWS:-0}"
MONITOR_TIMEOUT_SECONDS="${MONITOR_TIMEOUT_SECONDS:-900}"
LOG_DIR="$APP_DIR/logs"
LOG_FILE="$LOG_DIR/rebuild_from_2026-06-01.log"

cd "$APP_DIR"
mkdir -p "$LOG_DIR" data/recommendations data/backtests data/performance
: > "$LOG_FILE"
exec > >(tee -a "$LOG_FILE") 2>&1
export PYTHONUNBUFFERED=1

step() {
  echo ""
  echo "[$(date -Iseconds)] $1"
}

step "Rebuild started"
echo "Start date: $START_DATE"
echo "End date: $END_DATE"
echo "Windows: $WINDOWS"
echo "With news: $WITH_NEWS"
echo "Monitor timeout seconds: $MONITOR_TIMEOUT_SECONDS"

step "1/6 Checking Python files"
.venv/bin/python -m py_compile \
  ai_invest/strategy.py \
  ai_invest/optimizer.py \
  ai_invest/monitor.py \
  ai_invest/virtual_portfolio.py \
  ai_invest/cli.py

step "2/6 Optimizing strategy by 12/18/24 month windows"
.venv/bin/python main.py optimize-strategy --windows "$WINDOWS"

step "3/6 Rebuilding recommendations from $START_DATE to $END_DATE"
find data/recommendations -name "recommendations_2026-06-*.csv" -delete
if [ "$WITH_NEWS" = "1" ]; then
  .venv/bin/python main.py backfill-recommendations --start "$START_DATE" --end "$END_DATE" --with-news
else
  .venv/bin/python main.py backfill-recommendations --start "$START_DATE" --end "$END_DATE"
fi

step "4/6 Monitoring strategy and auto-optimizing if needed"
echo "Auto-monitor can run another optimization pass. It will continue without blocking after ${MONITOR_TIMEOUT_SECONDS}s."
if ! timeout "$MONITOR_TIMEOUT_SECONDS" .venv/bin/python main.py monitor-strategy --auto-optimize; then
  echo "Monitor auto-optimization timed out or failed. Saving monitor status without another optimization pass."
  .venv/bin/python main.py monitor-strategy || true
fi

step "5/6 Saving strategy status"
.venv/bin/python main.py strategy-status | tee "$LOG_DIR/strategy_status.txt"

step "6/6 Restarting web service"
sudo systemctl restart ai-invest

step "Rebuild finished"

#!/usr/bin/env bash
set -euo pipefail

APP_DIR="/home/ubuntu/ai_win_invest"

sudo apt update
sudo apt install -y python3 python3-venv python3-pip git nginx

cd "$APP_DIR"
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

sudo cp deploy/ai-invest.service /etc/systemd/system/ai-invest.service
sudo cp deploy/nginx-ai-invest.conf /etc/nginx/sites-available/ai-invest
sudo ln -sf /etc/nginx/sites-available/ai-invest /etc/nginx/sites-enabled/ai-invest
sudo rm -f /etc/nginx/sites-enabled/default

sudo nginx -t
sudo systemctl daemon-reload
sudo systemctl enable ai-invest
sudo systemctl restart ai-invest
sudo systemctl restart nginx

echo "AI Invest web is ready. Open http://52.62.125.23"

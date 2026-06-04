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
sudo cp deploy/ai-invest-daily.service /etc/systemd/system/ai-invest-daily.service
sudo cp deploy/ai-invest-daily.timer /etc/systemd/system/ai-invest-daily.timer
sudo cp deploy/ai-invest-weekly-optimize.service /etc/systemd/system/ai-invest-weekly-optimize.service
sudo cp deploy/ai-invest-weekly-optimize.timer /etc/systemd/system/ai-invest-weekly-optimize.timer
sudo cp deploy/nginx-ai-invest.conf /etc/nginx/sites-available/ai-invest
sudo ln -sf /etc/nginx/sites-available/ai-invest /etc/nginx/sites-enabled/ai-invest
sudo rm -f /etc/nginx/sites-enabled/default

sudo nginx -t
sudo systemctl daemon-reload
sudo systemctl enable ai-invest
sudo systemctl enable ai-invest-daily.timer
sudo systemctl enable ai-invest-weekly-optimize.timer
sudo systemctl restart ai-invest
sudo systemctl restart ai-invest-daily.timer
sudo systemctl restart ai-invest-weekly-optimize.timer
sudo systemctl restart nginx

echo "AI Invest web is ready on http://$(curl -s http://169.254.169.254/latest/meta-data/public-ipv4 || echo YOUR_PUBLIC_IP)"

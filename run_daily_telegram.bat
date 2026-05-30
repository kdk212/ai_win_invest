@echo off
cd /d "%~dp0"
python main.py daily --with-news --send-telegram
pause

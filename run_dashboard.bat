@echo off
cd /d "%~dp0"
echo AI Invest Korea dashboard is starting...
echo Open this address in Internet Explorer:
echo http://127.0.0.1:8787
echo.
"C:\Users\kdk21\AppData\Local\Programs\Python\Python312\python.exe" main.py web --port 8787
pause

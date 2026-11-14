@echo off
title EdgeVision Traffic Monitoring Server
echo ============================================================
echo Starting EdgeVision Offline Traffic Video Analytics Server
echo ============================================================
echo Opening http://127.0.0.1:5000 in your web browser...
cd /d "%~dp0"
python -m app.server
pause

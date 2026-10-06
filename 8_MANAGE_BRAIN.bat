@echo off
chcp 65001 > nul
cd /d "%~dp0"
title Polytrack Drosophila - Управление весами мозга (Brain Manager)

echo ===================================================================
echo     УПРАВЛЕНИЕ ВЕСАМИ МОЗГА ДРОЗОФИЛЫ (Polytrack Brain Manager)
echo ===================================================================
echo.

python brain_manager.py

echo.
pause

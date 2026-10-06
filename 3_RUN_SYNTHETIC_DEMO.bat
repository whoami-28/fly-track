@echo off
chcp 65001 > nul
cd /d "%~dp0"
title Polytrack Drosophila - Synthetic Simulation Demo

echo ===================================================================
echo   ДЕМОНСТРАЦИОННАЯ СИМУЛЯЦИЯ (Синтетический генератор Polytrack)
echo ===================================================================
echo.
echo Этот режим работает без запущенной игры Polytrack.
echo Симулируется видеопоток Polytrack, сетчатка мухи (HRC/EMD),
echo спайковый коннектом дрозофилы и приборная панель HUD.
echo.
echo Управление:
echo   [ПРОБЕЛ] - Пауза / Возобновление
echo   [Q/ESC]  - Выход
echo.

python main.py --synthetic --weights data/brain_weights_best.pt

echo.
pause

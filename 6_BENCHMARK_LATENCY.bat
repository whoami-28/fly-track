@echo off
chcp 65001 > nul
cd /d "%~dp0"
title Polytrack Drosophila - Latency & FPS Benchmark

echo ===================================================================
echo   БЕНЧМАРК ЗАДЕРЖКИ И ПРОПУСКНОЙ СПОСОБНОСТИ (Бюджет ^< 30-40 мс)
echo ===================================================================
echo.
echo Тестирование времени захвата экрана (mss), OpenCV и эмуляции WASD...
echo.

python game_harness.py --benchmark --cycles 200

echo.
pause

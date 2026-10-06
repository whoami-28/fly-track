@echo off
chcp 65001 > nul
cd /d "%~dp0"
title Polytrack Drosophila - Live Autonomous Pilot

echo ===================================================================
echo   ЗАПУСК АВТОНОМНОГО УПРАВЛЕНИЯ (Desktop Polytrack + FlyWire SNN)
echo ===================================================================
echo.
echo Убедитесь, что игра Polytrack запущена и видна на экране.
echo.
echo Горячие клавиши в окне HUD:
echo   [ПРОБЕЛ] - Пауза / Возобновление управления
echo   [M]      - Переключение Режима (Автопилот / Ручной)
echo   [R]      - Принудительный сброс трассы
echo   [Q/ESC]  - Безопасный выход
echo.
echo Запуск нейросети мухи с весами из data/brain_weights_best.pt...
echo.

python main.py --weights data/brain_weights_best.pt

echo.
echo Сессия управления завершена. Клавиши освобождены.
pause

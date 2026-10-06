@echo off
chcp 65001 > nul
cd /d "%~dp0"
title Polytrack Drosophila - Window Calibration

echo ===================================================================
echo     КАЛИБРОВКА ОКНА POLYTRACK (Drosophila Vision-Motor Loop)
echo ===================================================================
echo.
echo [1] Интерактивный выбор рамки окна (выделить область мышью)
echo [2] Автоматический поиск окна Polytrack по заголовку
echo [3] Предпросмотр захвата экрана в реальном времени (Preview)
echo.
set /p choice="Выберите действие [1/2/3] (по умолчанию 1): "

if "%choice%"=="2" (
    echo.
    echo Поиск окна Polytrack...
    python game_harness.py --autofind
) else if "%choice%"=="3" (
    echo.
    echo Запуск предпросмотра захвата...
    python game_harness.py --preview
) else (
    echo.
    echo Запуск интерактивного выделения окна игры...
    echo В появившемся окне выделите область игры Polytrack мышью,
    echo затем нажмите ENTER или SPACE для подтверждения.
    python game_harness.py --calibrate
)

echo.
pause

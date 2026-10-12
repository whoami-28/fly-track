@echo off
chcp 65001 > nul
cd /d "%~dp0"
title Polytrack Drosophila - Анализатор телеметрии (Flight Recorder)

echo ===================================================================
echo     DROSOPHILA FLIGHT RECORDER: ДИАГНОСТИКА И АНАЛИЗ ТЕЛЕМЕТРИИ
echo ===================================================================
echo.
echo [1] Последний заезд (сводка инцидента и траектория последних 30 кадров)
echo [2] Таблица всех заездов (master episodes summary)
echo [3] Выбрать конкретный CSV файл лога
echo [4] Выход
echo.
set /p choice="Выберите действие [1-4]: "

if "%choice%"=="1" goto latest
if "%choice%"=="2" goto summary
if "%choice%"=="3" goto file
if "%choice%"=="4" goto exit

:latest
echo.
python telemetry_logger.py --latest --tail 30
pause
goto exit

:summary
echo.
python telemetry_logger.py --summary
pause
goto exit

:file
echo.
set /p filepath="Введите путь к файлу CSV: "
python telemetry_logger.py --file "%filepath%" --tail 30
pause
goto exit

:exit

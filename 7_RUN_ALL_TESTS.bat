@echo off
chcp 65001 > nul
cd /d "%~dp0"
title Polytrack Drosophila - Запуск всех тестов

echo ===================================================================
echo       ЗАПУСК ПОЛНОГО КОМПЛЕКСА ТЕСТОВ СИСТЕМЫ (MODULES 1-5)
echo ===================================================================
echo.

echo [1/6] Тестирование захвата и эмулятора (game_harness)...
python test_harness.py
if errorlevel 1 goto failed

echo.
echo [2/6] Тестирование сетчатки и оптического потока (retina)...
python test_retina.py
if errorlevel 1 goto failed

echo.
echo [3/6] Тестирование графа коннектома FlyWire (connectome_loader)...
python test_connectome.py
if errorlevel 1 goto failed

echo.
echo [4/6] Тестирование спайкового мозга дрозофилы (fly_brain)...
python test_brain.py
if errorlevel 1 goto failed

echo.
echo [5/6] Тестирование менеджера весов и сброса мозга (brain_manager)...
python test_brain_manager.py
if errorlevel 1 goto failed

echo.
echo [6/6] Тестирование системного контура и тренера (system & trainer)...
python test_system.py
if errorlevel 1 goto failed

echo.
echo ===================================================================
echo       ВСЕ ТЕСТЫ ПРОЙДЕНЫ УСПЕШНО! СИСТЕМА ПОЛНОСТЬЮ ИСПРАВНА.
echo ===================================================================
goto end

:failed
echo.
echo ===================================================================
echo    ОШИБКА: ОБНАРУЖЕНЫ СБОИ В ОДНОМ ИЛИ НЕСКОЛЬКИХ МОДУЛЯХ!
echo ===================================================================

:end
echo.
pause

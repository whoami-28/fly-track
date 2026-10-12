@echo off
chcp 65001 > nul
cd /d "%~dp0"
title Polytrack Drosophila - Запуск всех тестов

echo ===================================================================
echo       ЗАПУСК ПОЛНОГО КОМПЛЕКСА ТЕСТОВ СИСТЕМЫ (MODULES 1-5)
echo ===================================================================
echo.

echo [1/8] Тестирование захвата и эмулятора (game_harness)...
python test_harness.py
if errorlevel 1 goto failed

echo.
echo [2/8] Тестирование сетчатки и оптического потока (retina)...
python test_retina.py
if errorlevel 1 goto failed

echo.
echo [3/8] Тестирование графа коннектома FlyWire (connectome_loader)...
python test_connectome.py
if errorlevel 1 goto failed

echo.
echo [4/8] Тестирование спайкового мозга дрозофилы (fly_brain)...
python test_brain.py
if errorlevel 1 goto failed

echo.
echo [5/8] Тестирование детектора границ и поребриков (boundary_detector)...
python test_boundary_detector.py
if errorlevel 1 goto failed

echo.
echo [6/8] Тестирование прозрачного оверлея коннектома (neural_overlay)...
python test_overlay.py
if errorlevel 1 goto failed

echo.
echo [7/8] Тестирование менеджера весов и сброса мозга (brain_manager)...
python test_brain_manager.py
if errorlevel 1 goto failed

echo.
echo [8/9] Тестирование системного контура и тренера (system & trainer)...
python test_system.py
if errorlevel 1 goto failed

echo.
echo [9/9] Тестирование бортового самописца телеметрии (telemetry_logger)...
python -m pytest test_telemetry.py -q
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

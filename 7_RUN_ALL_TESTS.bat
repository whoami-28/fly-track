@echo off
chcp 65001 > nul
cd /d "%~dp0"
title Polytrack Drosophila - System Test Suite

echo ===================================================================
echo     ЗАПУСК ПОЛНОГО НАБОРА ТЕСТОВ (Все модули контура мухи)
echo ===================================================================
echo.
echo [1/5] Тестирование модуля захвата и детекции (test_harness.py)...
python test_harness.py
if errorlevel 1 goto error

echo.
echo [2/5] Тестирование сетчатки и оптического потока (test_retina.py)...
python test_retina.py
if errorlevel 1 goto error

echo.
echo [3/5] Тестирование загрузчика коннектома FlyWire (test_connectome.py)...
python test_connectome.py
if errorlevel 1 goto error

echo.
echo [4/5] Тестирование спайковой нейросети LIF (test_brain.py)...
python test_brain.py
if errorlevel 1 goto error

echo.
echo [5/5] Интеграционное тестирование всей системы (test_system.py)...
python test_system.py
if errorlevel 1 goto error

echo.
echo ===================================================================
echo        ВСЕ ТЕСТЫ УСПЕШНО ПРОЙДЕНЫ! (100%% PASS)
echo ===================================================================
goto end

:error
echo.
echo !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
echo                     ОБНАРУЖЕНА ОШИБКА В ТЕСТАХ
echo !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

:end
echo.
pause

@echo off
chcp 65001 > nul
cd /d "%~dp0"
title Polytrack Drosophila - Retina Diagnostic Dashboard

echo ===================================================================
echo     ДИАГНОСТИЧЕСКАЯ ПАНЕЛЬ СЕТЧАТКИ ДРОЗОФИЛЫ (LIVE RETINA)
echo ===================================================================
echo.
echo Режим: ПРЯМОЙ ЗАХВАТ ОКНА POLYTRACK (РЕАЛЬНОЕ ЗРЕНИЕ МУХИ)
echo   - 512 бинокулярных омматидиев
echo   - Выделение оптического потока (корреляторы Хассенштейна-Райхардта)
echo   - Каналы ON / OFF
echo   - Сигналы LPTC (HS_L, HS_R, VS_Forward, Yaw)
echo.
echo Нажмите 'q' или ESC в окне дашборда для выхода.
echo Нажмите 's' для сохранения снимка в папку data/
echo.

python retina.py --stream --live-screen

echo.
pause

@echo off
chcp 65001 > nul
cd /d "%~dp0"
title Polytrack Drosophila - Retina Diagnostic Dashboard

echo ===================================================================
echo     ДИАГНОСТИЧЕСКАЯ ПАНЕЛЬ СЕТЧАТКИ ДРОЗОФИЛЫ (RETINA HRC)
echo ===================================================================
echo.
echo Открытие визуализатора сетчатки:
echo   - 512 бинокулярных омматидиев
echo   - Выделение оптического потока (корреляторы Хассенштейна-Райхардта)
echo   - Каналы ON / OFF
echo   - Сигналы LPTC (HS_L, HS_R, VS_Forward, Yaw)
echo.
echo Нажмите 'q' или ESC в окне дашборда для выхода.
echo.

python retina.py --stream

echo.
pause

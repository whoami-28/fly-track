@echo off
chcp 65001 > nul
cd /d "%~dp0"
title Polytrack Drosophila - Автопоиск окна PolyTrack

echo ===================================================================
echo     АВТОМАТИЧЕСКИЙ ПОИСК ОКНА POLYTRACK (Drosophila Game Harness)
echo ===================================================================
echo.
echo Поиск активного окна игры PolyTrack...
echo.

python game_harness.py --autofind
if errorlevel 1 goto NOT_FOUND

echo.
echo ===================================================================
echo  [УСПЕХ] Окно PolyTrack успешно найдено и откалибровано!
echo  Координаты сохранены в calibration_config.json
echo ===================================================================
goto END

:NOT_FOUND
echo.
echo ===================================================================
echo  [ОШИБКА] Окно PolyTrack не найдено.
echo  1. Запустите игру PolyTrack.
echo  2. Убедитесь, что окно не свернуто в панель задач.
echo  3. Запустите этот скрипт еще раз.
echo ===================================================================

:END
echo.
pause

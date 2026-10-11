@echo off
chcp 65001 > nul
cd /d "%~dp0"
title Polytrack Drosophila - Evolutionary Training

echo ===================================================================
echo     ЭВОЛЮЦИОННОЕ ОБУЧЕНИЕ МУХИ (Элитный алгоритм Best Point)
echo ===================================================================
echo.
echo Все поколения надежно сохраняют лучшего чемпиона (Best Point).
echo При обнаружении плашки респауна ("Нажмите R / Enter...")
echo попытка немедленно прерывается и начинается следующий заезд.
echo.
echo [1] Обучение на живой игре Desktop Polytrack (требуется окно игры)
echo [2] Быстрое обучение на синтетическом треке (фоновый режим)
echo.
set /p mode="Выберите режим [1/2] (по умолчанию 1): "

set /p gens="Количество поколений [по умолчанию 10]: "
if "%gens%"=="" set gens=10

set /p pop="Размер популяции [по умолчанию 8]: "
if "%pop%"=="" set pop=8

set /p frames="Длительность заезда в кадрах [по умолчанию 900 (~20 сек + овертайм)]: "
if "%frames%"=="" set frames=900

if "%mode%"=="2" (
    echo.
    echo Запуск синтетического обучения: %gens% поколений, %pop% кандидатов, %frames% кадров...
    python trainer.py --generations %gens% --population %pop% --max-frames %frames%
) else (
    echo.
    echo Запуск обучения на живой игре Polytrack: %gens% поколений, %pop% кандидатов, %frames% кадров...
    echo Переключитесь на окно игры Polytrack!
    python trainer.py --live --generations %gens% --population %pop% --max-frames %frames%
)

echo.
echo Обучение завершено. Лучшие веса сохранены в data/brain_weights_best.pt
pause

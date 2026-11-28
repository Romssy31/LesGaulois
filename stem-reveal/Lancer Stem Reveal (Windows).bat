@echo off
chcp 65001 >nul
title Stem Reveal
cd /d "%~dp0"

set "PY="
where py >nul 2>nul && set "PY=py -3"
if not defined PY where python >nul 2>nul && set "PY=python"
if not defined PY goto nopython

if not exist ".venv\installed.ok" (
  echo.
  echo  Premiere installation : 5 a 10 minutes, une seule fois.
  echo.
  if exist ".venv" rmdir /s /q ".venv"
  %PY% -m venv .venv || goto nopython
  ".venv\Scripts\python.exe" -m pip install --upgrade pip
  ".venv\Scripts\python.exe" -m pip install -r requirements.txt || goto fail
  echo ok> ".venv\installed.ok"
  copy /y requirements.txt ".venv\requirements.installed" >nul
)

fc /b requirements.txt ".venv\requirements.installed" >nul 2>nul
if errorlevel 1 (
  echo  Mise a jour des composants...
  ".venv\Scripts\python.exe" -m pip install -r requirements.txt || goto fail
  copy /y requirements.txt ".venv\requirements.installed" >nul
)

".venv\Scripts\python.exe" app.py
pause
exit /b

:nopython
echo.
echo  Python n'est pas installe.
echo  Installe Python 3.12 depuis https://www.python.org/downloads/
echo  et coche "Add Python to PATH" pendant l'installation, puis relance ce fichier.
start https://www.python.org/downloads/
pause
exit /b

:fail
echo.
echo  L'installation a echoue (verifie ta connexion internet) puis relance ce fichier.
pause
exit /b

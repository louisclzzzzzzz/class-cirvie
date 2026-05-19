@echo off
REM ============================================================
REM  Build predict_server.exe pour distribution Windows
REM  Exécuter depuis la racine du projet (où se trouvent les .pkl)
REM ============================================================

REM -- Détection de Python ------------------------------------------
SET PYTHON=
WHERE py >nul 2>&1 && SET PYTHON=py
IF "%PYTHON%"=="" WHERE python >nul 2>&1 && SET PYTHON=python
IF "%PYTHON%"=="" WHERE python3 >nul 2>&1 && SET PYTHON=python3

IF "%PYTHON%"=="" (
    echo.
    echo [ERREUR] Python est introuvable sur ce poste.
    echo.
    echo  1. Telechargez Python 3.11+ depuis https://www.python.org/downloads/
    echo  2. COCHEZ "Add Python to PATH" lors de l'installation
    echo  3. Relancez ce script
    echo.
    pause
    exit /b 1
)

echo [OK] Python detecte : %PYTHON%
%PYTHON% --version

REM -- Installation des dependances ---------------------------------
echo.
echo Installation de PyInstaller et Flask...
%PYTHON% -m pip install --quiet --upgrade pyinstaller flask

IF ERRORLEVEL 1 (
    echo [ERREUR] pip a echoue. Verifiez votre connexion internet.
    pause
    exit /b 1
)

REM -- Compilation --------------------------------------------------
echo.
echo Compilation en cours (peut prendre 1-2 minutes)...
%PYTHON% -m PyInstaller --onefile --noconsole ^
  --name predict_server ^
  --add-data "model_fast_omv.pkl;." ^
  --add-data "model_fast_service.pkl;." ^
  --add-data "model_fast_origine.pkl;." ^
  --add-data "nom_service.json;." ^
  predict_server.py

IF ERRORLEVEL 1 (
    echo.
    echo [ERREUR] La compilation a echoue. Verifiez les messages ci-dessus.
    pause
    exit /b 1
)

echo.
echo [OK] Exe genere : dist\predict_server.exe
echo.
echo Contenu du zip de livraison :
echo   dist\predict_server.exe
echo   vba\modPredict.bas  (a importer dans ClasseurCIRVIE.xlsm)
echo   LISEZMOI.txt
pause

@echo off
REM ============================================================
REM  Build predict_server.exe pour distribution Windows
REM  Exécuter depuis la racine du projet (où se trouvent les .pkl)
REM ============================================================

pip install --quiet pyinstaller flask

pyinstaller --onefile --noconsole ^
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
echo   ClasseurCIRVIE.xlsm
echo   LISEZMOI.txt
pause

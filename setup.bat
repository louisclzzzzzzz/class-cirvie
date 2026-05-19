@echo off
REM ============================================================
REM  Setup CIRVIE — a executer UNE SEULE FOIS apres git clone
REM  Cree ClasseurCIRVIE.xlsm avec les macros integrees
REM ============================================================

IF EXIST ClasseurCIRVIE.xlsm (
    echo [OK] ClasseurCIRVIE.xlsm existe deja. Rien a faire.
    echo      Pour recreer le classeur, supprimez-le d'abord.
    pause
    exit /b 0
)

IF NOT EXIST predict_server.exe (
    echo [ATTENTION] predict_server.exe manquant.
    echo             Faites un "git pull" pour recuperer la derniere version.
    pause
)

echo Creation de ClasseurCIRVIE.xlsm...
powershell -ExecutionPolicy Bypass -File "%~dp0setup_excel.ps1" "%~dp0"

IF ERRORLEVEL 1 (
    echo.
    echo [ERREUR] La creation du classeur a echoue.
    echo.
    echo  Solution alternative :
    echo  1. Ouvrez Excel
    echo  2. Creez un nouveau classeur
    echo  3. Appuyez sur ALT+F11 pour ouvrir l'editeur VBA
    echo  4. Fichier > Importer un fichier > selectionnez vba\modPredict.bas
    echo  5. Sauvegardez sous le nom "ClasseurCIRVIE.xlsm" dans ce dossier
    echo.
    pause
    exit /b 1
)

echo.
echo [OK] ClasseurCIRVIE.xlsm cree avec succes.
echo.
echo Pour utiliser :
echo  1. Ouvrez ClasseurCIRVIE.xlsm
echo  2. Activez les macros si demande
echo  3. Remplissez les colonnes A a E et cliquez sur "Classifier"
echo.
pause

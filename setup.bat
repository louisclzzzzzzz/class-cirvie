@echo off
REM ============================================================
REM  Setup CIRVIE — a executer UNE SEULE FOIS apres git clone
REM  Convertit CIRVIE_INCIDENTS_2026.xlsx en .xlsm avec macros
REM ============================================================

IF EXIST CIRVIE_INCIDENTS_2026.xlsm (
    echo [OK] CIRVIE_INCIDENTS_2026.xlsm existe deja. Rien a faire.
    echo      Pour recreer le fichier, supprimez CIRVIE_INCIDENTS_2026.xlsm d'abord.
    pause
    exit /b 0
)

IF NOT EXIST CIRVIE_INCIDENTS_2026.xlsx (
    echo [ERREUR] CIRVIE_INCIDENTS_2026.xlsx introuvable.
    echo          Faites un "git pull" pour recuperer le fichier.
    pause
    exit /b 1
)

IF NOT EXIST predict_server.exe (
    echo [ATTENTION] predict_server.exe manquant.
    echo             Faites un "git pull" pour recuperer la derniere version.
    echo             Vous pourrez relancer ce script apres.
    pause
)

echo Conversion de CIRVIE_INCIDENTS_2026.xlsx en .xlsm avec macros...
powershell -ExecutionPolicy Bypass -File "%~dp0setup_excel.ps1" "%~dp0"

IF ERRORLEVEL 1 (
    echo.
    echo [ERREUR] La conversion automatique a echoue.
    echo.
    echo  Procedure manuelle :
    echo  1. Ouvrez CIRVIE_INCIDENTS_2026.xlsx avec Excel
    echo  2. Appuyez sur ALT+F11
    echo  3. Fichier > Importer un fichier > selectionnez vba\modPredict.bas
    echo  4. Fermez l'editeur VBA
    echo  5. Fichier > Enregistrer sous > choisir "Classeur Excel active les macros (*.xlsm)"
    echo     Nom du fichier : CIRVIE_INCIDENTS_2026.xlsm
    echo.
    pause
    exit /b 1
)

echo.
echo [OK] Setup termine.
echo.
echo  Ouvrez CIRVIE_INCIDENTS_2026.xlsm
echo  Activez les macros si demande
echo  Utilisez les boutons "Classifier" sur la feuille "TOUS LES INCIDENTS"
echo.
pause

# setup_excel.ps1
# Convertit CIRVIE_INCIDENTS_2026.xlsx en .xlsm et injecte les macros VBA
param([string]$RepoDir = $PSScriptRoot)

$xlsxPath = Join-Path $RepoDir "CIRVIE_INCIDENTS_2026.xlsx"
$xlsmPath = Join-Path $RepoDir "CIRVIE_INCIDENTS_2026.xlsm"
$basPath   = Join-Path $RepoDir "vba\modPredict.bas"

if (-not (Test-Path $xlsxPath)) {
    Write-Error "Fichier source introuvable : $xlsxPath"
    exit 1
}
if (-not (Test-Path $basPath)) {
    Write-Error "Module VBA introuvable : $basPath"
    exit 1
}

try {
    $excel = New-Object -ComObject Excel.Application
} catch {
    Write-Error "Excel n'est pas accessible via COM."
    exit 1
}

$excel.Visible       = $false
$excel.DisplayAlerts = $false

$wb = $excel.Workbooks.Open($xlsxPath)

# Import du module VBA
$imported = $false
try {
    $wb.VBProject.VBComponents.Import($basPath) | Out-Null
    $imported = $true
    Write-Host "  [OK] Module VBA importe."
} catch {
    Write-Warning "Import VBA automatique impossible (acces au projet VBA restreint par la politique IT)."
}

# Ajout du bouton "Classifier" sur la feuille "TOUS LES INCIDENTS" (si VBA importé)
if ($imported) {
    try {
        $ws = $wb.Worksheets.Item("TOUS LES INCIDENTS")

        # Bouton ligne active
        $btn1 = $ws.Buttons.Add(0, 0, 130, 22)
        $btn1.Caption  = "Classifier cette ligne"
        $btn1.OnAction = "ClassifierLigne"
        $btn1.Placement = 3   # xlFreeFloating

        # Bouton lignes vides
        $btn2 = $ws.Buttons.Add(140, 0, 160, 22)
        $btn2.Caption  = "Classifier lignes vides"
        $btn2.OnAction = "ClassifierLignesVides"
        $btn2.Placement = 3

        Write-Host "  [OK] Boutons ajoutes sur la feuille 'TOUS LES INCIDENTS'."
    } catch {
        Write-Warning "Impossible d'ajouter les boutons : $_"
    }
}

# Sauvegarde en .xlsm (52 = xlOpenXMLWorkbookMacroEnabled)
$wb.SaveAs($xlsmPath, 52)
$wb.Close($false)
$excel.Quit()
[System.Runtime.InteropServices.Marshal]::ReleaseComObject($excel) | Out-Null

Write-Host ""
Write-Host "  [OK] Fichier sauvegarde : $xlsmPath"

if (-not $imported) {
    Write-Host ""
    Write-Host "  ACTION MANUELLE REQUISE :"
    Write-Host "  1. Ouvrez CIRVIE_INCIDENTS_2026.xlsm"
    Write-Host "  2. Appuyez sur ALT+F11 (editeur VBA)"
    Write-Host "  3. Fichier > Importer un fichier > selectionnez : vba\modPredict.bas"
    Write-Host "  4. Fermez l'editeur et sauvegardez (Ctrl+S)"
}
exit 0

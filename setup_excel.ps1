# setup_excel.ps1 — Cree ClasseurCIRVIE.xlsm avec macros VBA
param([string]$RepoDir = $PSScriptRoot)

$xlsmPath = Join-Path $RepoDir "ClasseurCIRVIE.xlsm"
$basPath   = Join-Path $RepoDir "vba\modPredict.bas"

if (-not (Test-Path $basPath)) {
    Write-Error "Fichier VBA introuvable : $basPath"
    exit 1
}

try {
    $excel = New-Object -ComObject Excel.Application
} catch {
    Write-Error "Excel n'est pas installe ou accessible via COM."
    exit 1
}

$excel.Visible        = $false
$excel.DisplayAlerts  = $false

$wb = $excel.Workbooks.Add()
$ws = $wb.Worksheets.Item(1)
$ws.Name = "Saisie"

# En-têtes
$headers = @("Description","Demandeur","Cause","Traite par","Urgence","OMV","SERVICE","ORIGINE")
for ($i = 0; $i -lt $headers.Length; $i++) {
    $cell = $ws.Cells.Item(1, $i + 1)
    $cell.Value2 = $headers[$i]
    $cell.Font.Bold = $true
}

# Mise en forme colonnes résultats (F, G, H)
$ws.Columns.Item(1).ColumnWidth = 60   # Description
$ws.Columns.Item(6).Interior.Color = 0xD9EAD3   # OMV vert clair
$ws.Columns.Item(7).Interior.Color = 0xCFE2F3   # SERVICE bleu clair
$ws.Columns.Item(8).Interior.Color = 0xFFF2CC   # ORIGINE jaune clair

# Ligne d'exemple
$ws.Cells.Item(2, 1).Value2 = "Rachat partiel sur le contrat 30421. Courrier non envoye."
$ws.Cells.Item(2, 2).Value2 = "DUPONT, Marie"
$ws.Cells.Item(2, 3).Value2 = "BASE DE DONNEES"
$ws.Cells.Item(2, 4).Value2 = "MOA OMVIE"
$ws.Cells.Item(2, 5).Value2 = "2"

# Import du module VBA
$imported = $false
try {
    # Nécessite "Faire confiance au modele d'objet du projet VBA"
    # dans Fichier > Options > Centre de gestion de la confidentialite > Parametres des macros
    $wb.VBProject.VBComponents.Import($basPath) | Out-Null
    $imported = $true
    Write-Host "  [OK] Module VBA importe."
} catch {
    Write-Warning "Import VBA automatique impossible (acces au projet VBA restreint)."
    Write-Warning "Etape manuelle : ALT+F11 > Fichier > Importer > vba\modPredict.bas"
}

# Bouton "Classifier" (uniquement si VBA importé)
if ($imported) {
    $btn = $ws.Buttons.Add(480, 5, 120, 25)
    $btn.Caption  = "Classifier"
    $btn.OnAction = "ClassifierIncident"
}

# Sauvegarde en .xlsm (52 = xlOpenXMLWorkbookMacroEnabled)
$wb.SaveAs($xlsmPath, 52)
$wb.Close($false)
$excel.Quit()
[System.Runtime.InteropServices.Marshal]::ReleaseComObject($excel) | Out-Null

Write-Host "  [OK] Classeur sauvegarde : $xlsmPath"
if (-not $imported) {
    Write-Host ""
    Write-Host "  ACTION REQUISE : importez le module VBA manuellement."
    Write-Host "  1. Ouvrez ClasseurCIRVIE.xlsm"
    Write-Host "  2. ALT+F11 > Fichier > Importer un fichier"
    Write-Host "  3. Selectionnez : vba\modPredict.bas"
    Write-Host "  4. Sauvegardez (Ctrl+S)"
}
exit 0

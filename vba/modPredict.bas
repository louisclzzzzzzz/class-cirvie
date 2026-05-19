Attribute VB_Name = "modPredict"
' ============================================================
'  Module VBA — Classification CIRVIE
'  Feuille cible : "TOUS LES INCIDENTS"
'
'  Colonnes lues (input) :
'    H  (8)  : Description
'    I  (9)  : Demandeur
'    J  (10) : Urgence
'    P  (16) : Traite par
'    Q  (17) : Cause reelle
'
'  Colonnes ecrites (output) :
'    E  (5)  : INTERVENTION OMV
'    F  (6)  : SERVICE
'    G  (7)  : ORIGINE
' ============================================================
Option Explicit

Private Const SERVER_URL      As String = "http://localhost:8765"
Private Const START_TIMEOUT_S As Long   = 15
Private Const SHEET_NAME      As String = "TOUS LES INCIDENTS"
Private Const FIRST_DATA_ROW  As Long   = 2   ' ligne 1 = en-têtes

' Colonnes input
Private Const COL_DESCRIPTION As Long = 8   ' H
Private Const COL_DEMANDEUR   As Long = 9   ' I
Private Const COL_URGENCE     As Long = 10  ' J
Private Const COL_TRAITE      As Long = 16  ' P
Private Const COL_CAUSE       As Long = 17  ' Q

' Colonnes output
Private Const COL_OMV         As Long = 5   ' E
Private Const COL_SERVICE     As Long = 6   ' F
Private Const COL_ORIGINE     As Long = 7   ' G


' ------------------------------------------------------------
'  Macro principale : classifier la ligne active
' ------------------------------------------------------------
Public Sub ClassifierLigne()
    Dim ws As Worksheet
    Set ws = GetIncidentSheet()
    If ws Is Nothing Then Exit Sub

    If Not EnsureServer() Then Exit Sub

    Dim r As Long
    r = ActiveCell.Row
    If r < FIRST_DATA_ROW Then
        MsgBox "Placez le curseur sur une ligne de donnees (pas l'en-tete).", vbInformation, "CIRVIE"
        Exit Sub
    End If

    ClassifierRow ws, r
End Sub


' ------------------------------------------------------------
'  Macro : classifier toutes les lignes sans OMV rempli
' ------------------------------------------------------------
Public Sub ClassifierLignesVides()
    Dim ws As Worksheet
    Set ws = GetIncidentSheet()
    If ws Is Nothing Then Exit Sub

    If Not EnsureServer() Then Exit Sub

    Dim lastRow As Long
    lastRow = ws.Cells(ws.Rows.Count, COL_DESCRIPTION).End(xlUp).Row

    Dim count As Long
    count = 0
    Dim r As Long
    For r = FIRST_DATA_ROW To lastRow
        If Trim(CStr(ws.Cells(r, COL_OMV).Value)) = "" Then
            If Trim(CStr(ws.Cells(r, COL_DESCRIPTION).Value)) <> "" Then
                ClassifierRow ws, r
                count = count + 1
            End If
        End If
    Next r

    MsgBox count & " ligne(s) classifiee(s).", vbInformation, "CIRVIE"
End Sub


' ------------------------------------------------------------
'  Macro : classifier toutes les lignes (avec ou sans OMV)
' ------------------------------------------------------------
Public Sub ClassifierToutes()
    Dim ws As Worksheet
    Set ws = GetIncidentSheet()
    If ws Is Nothing Then Exit Sub

    If Not EnsureServer() Then Exit Sub

    Dim lastRow As Long
    lastRow = ws.Cells(ws.Rows.Count, COL_DESCRIPTION).End(xlUp).Row

    Dim count As Long
    count = 0
    Dim r As Long
    For r = FIRST_DATA_ROW To lastRow
        If Trim(CStr(ws.Cells(r, COL_DESCRIPTION).Value)) <> "" Then
            ClassifierRow ws, r
            count = count + 1
        End If
    Next r

    MsgBox count & " ligne(s) classifiee(s).", vbInformation, "CIRVIE"
End Sub


' ============================================================
'  Cœur : classifier une ligne et ecrire les resultats
' ============================================================
Private Sub ClassifierRow(ws As Worksheet, r As Long)
    Dim desc    As String: desc    = EscapeJson(CStr(ws.Cells(r, COL_DESCRIPTION).Value))
    Dim demand  As String: demand  = EscapeJson(CStr(ws.Cells(r, COL_DEMANDEUR).Value))
    Dim urgence As String: urgence = EscapeJson(CStr(ws.Cells(r, COL_URGENCE).Value))
    Dim traite  As String: traite  = EscapeJson(CStr(ws.Cells(r, COL_TRAITE).Value))
    Dim cause   As String: cause   = EscapeJson(CStr(ws.Cells(r, COL_CAUSE).Value))

    Dim body As String
    body = "{" & _
           """description"":""" & desc    & """," & _
           """demandeur"":"""   & demand  & """," & _
           """urgence"":"""     & urgence & """," & _
           """traite"":"""      & traite  & """," & _
           """cause"":"""       & cause   & """" & _
           "}"

    Dim resp As String
    resp = HttpPost(SERVER_URL & "/predict", body)

    If resp = "" Then
        ws.Cells(r, COL_OMV).Value     = "ERREUR"
        ws.Cells(r, COL_SERVICE).Value = "ERREUR"
        ws.Cells(r, COL_ORIGINE).Value = "ERREUR"
        Exit Sub
    End If

    ws.Cells(r, COL_OMV).Value     = ExtractPrediction(resp, "omv")
    ws.Cells(r, COL_SERVICE).Value = ExtractPrediction(resp, "service")
    ws.Cells(r, COL_ORIGINE).Value = ExtractPrediction(resp, "origine")
End Sub


' ============================================================
'  Helpers HTTP
' ============================================================

Private Function EnsureServer() As Boolean
    If ServerRunning() Then
        EnsureServer = True
        Exit Function
    End If
    StartServer
    If Not ServerRunning() Then
        MsgBox "Impossible de demarrer predict_server.exe." & vbCrLf & _
               "Verifiez qu'il est bien dans le meme dossier que ce classeur.", _
               vbCritical, "CIRVIE"
        EnsureServer = False
    Else
        EnsureServer = True
    End If
End Function


Private Function GetIncidentSheet() As Worksheet
    On Error Resume Next
    Dim ws As Worksheet
    Set ws = ThisWorkbook.Worksheets(SHEET_NAME)
    On Error GoTo 0
    If ws Is Nothing Then
        MsgBox "Feuille '" & SHEET_NAME & "' introuvable dans ce classeur.", vbCritical, "CIRVIE"
    End If
    Set GetIncidentSheet = ws
End Function


Private Function ServerRunning() As Boolean
    On Error Resume Next
    Dim http As Object
    Set http = CreateObject("WinHttp.WinHttpRequest.5.1")
    http.SetTimeouts 1000, 1000, 1000, 1000
    http.Open "GET", SERVER_URL & "/health", False
    http.Send
    ServerRunning = (http.Status = 200)
    On Error GoTo 0
End Function


Private Sub StartServer()
    Dim exePath As String
    exePath = ThisWorkbook.Path & "\predict_server.exe"

    If Dir(exePath) = "" Then
        MsgBox "predict_server.exe introuvable dans :" & vbCrLf & ThisWorkbook.Path, _
               vbCritical, "CIRVIE"
        Exit Sub
    End If

    Shell """" & exePath & """", vbHide

    Dim t0 As Single
    t0 = Timer
    Do While Not ServerRunning()
        If Timer - t0 > START_TIMEOUT_S Then Exit Do
        Application.Wait Now + TimeValue("00:00:01")
    Loop
End Sub


Private Function HttpPost(url As String, body As String) As String
    On Error Resume Next
    Dim http As Object
    Set http = CreateObject("WinHttp.WinHttpRequest.5.1")
    http.SetTimeouts 5000, 5000, 5000, 30000
    http.Open "POST", url, False
    http.SetRequestHeader "Content-Type", "application/json"
    http.Send body
    If http.Status = 200 Then
        HttpPost = http.ResponseText
    Else
        HttpPost = ""
    End If
    On Error GoTo 0
End Function


' ============================================================
'  Parseur JSON minimal
' ============================================================
Private Function ExtractPrediction(jsonStr As String, bloc As String) As String
    Dim startBloc As Long
    startBloc = InStr(1, jsonStr, """" & bloc & """", vbTextCompare)
    If startBloc = 0 Then
        ExtractPrediction = ""
        Exit Function
    End If

    Dim re As Object
    Set re = CreateObject("VBScript.RegExp")
    re.Pattern = """prediction""\s*:\s*""([^""]+)"""
    re.IgnoreCase = True

    Dim m As Object
    Set m = re.Execute(Mid(jsonStr, startBloc))
    If m.Count > 0 Then
        ExtractPrediction = m.Item(0).SubMatches(0)
    Else
        ExtractPrediction = ""
    End If
End Function


Private Function EscapeJson(s As String) As String
    s = Replace(s, "\",  "\\")
    s = Replace(s, """", "\""")
    s = Replace(s, Chr(8),  "\b")
    s = Replace(s, Chr(9),  "\t")
    s = Replace(s, Chr(10), "\n")
    s = Replace(s, Chr(13), "\r")
    EscapeJson = s
End Function

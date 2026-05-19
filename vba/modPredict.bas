Attribute VB_Name = "modPredict"
' ============================================================
'  Module VBA — Classification CIRVIE
'  Appelle predict_server.exe via HTTP localhost:8765
'
'  Colonnes attendues sur la feuille active :
'    A : Description
'    B : Demandeur
'    C : Cause
'    D : Traite par
'    E : Urgence
'    F : -> OMV      (ecrit par la macro)
'    G : -> SERVICE  (ecrit par la macro)
'    H : -> ORIGINE  (ecrit par la macro)
' ============================================================
Option Explicit

Private Const SERVER_URL As String = "http://localhost:8765"
Private Const SERVER_PORT As Long = 8765
Private Const START_TIMEOUT_S As Long = 15  ' secondes max pour demarrer le serveur


' ------------------------------------------------------------
'  Macro principale : classifier la ligne active
' ------------------------------------------------------------
Public Sub ClassifierIncident()
    Dim ws As Worksheet
    Set ws = ActiveSheet

    If Not ServerRunning() Then
        StartServer
        If Not ServerRunning() Then
            MsgBox "Impossible de demarrer predict_server.exe." & vbCrLf & _
                   "Verifiez que le fichier est bien dans le meme dossier que ce classeur.", _
                   vbCritical, "Erreur serveur CIRVIE"
            Exit Sub
        End If
    End If

    Dim r As Long
    r = ActiveCell.Row

    Dim desc    As String: desc    = EscapeJson(CStr(ws.Cells(r, 1).Value))
    Dim demand  As String: demand  = EscapeJson(CStr(ws.Cells(r, 2).Value))
    Dim cause   As String: cause   = EscapeJson(CStr(ws.Cells(r, 3).Value))
    Dim traite  As String: traite  = EscapeJson(CStr(ws.Cells(r, 4).Value))
    Dim urgence As String: urgence = EscapeJson(CStr(ws.Cells(r, 5).Value))

    Dim body As String
    body = "{" & _
           """description"":""" & desc    & """," & _
           """demandeur"":"""   & demand  & """," & _
           """cause"":"""       & cause   & """," & _
           """traite"":"""      & traite  & """," & _
           """urgence"":"""     & urgence & """" & _
           "}"

    Dim resp As String
    resp = HttpPost(SERVER_URL & "/predict", body)

    If resp = "" Then
        MsgBox "Le serveur n'a pas repondu. Relancez la macro.", vbExclamation, "CIRVIE"
        Exit Sub
    End If

    ws.Cells(r, 6).Value = ExtractPrediction(resp, "omv")
    ws.Cells(r, 7).Value = ExtractPrediction(resp, "service")
    ws.Cells(r, 8).Value = ExtractPrediction(resp, "origine")
End Sub


' ------------------------------------------------------------
'  Macro : classifier toutes les lignes d'une plage selectionnee
' ------------------------------------------------------------
Public Sub ClassifierPlage()
    If Not ServerRunning() Then
        StartServer
        If Not ServerRunning() Then
            MsgBox "Impossible de demarrer predict_server.exe.", vbCritical, "CIRVIE"
            Exit Sub
        End If
    End If

    Dim ws As Worksheet
    Set ws = ActiveSheet

    Dim firstRow As Long, lastRow As Long
    firstRow = Selection.Row
    lastRow  = firstRow + Selection.Rows.Count - 1

    Dim r As Long
    For r = firstRow To lastRow
        If Trim(CStr(ws.Cells(r, 1).Value)) = "" Then GoTo Suivant

        Dim desc    As String: desc    = EscapeJson(CStr(ws.Cells(r, 1).Value))
        Dim demand  As String: demand  = EscapeJson(CStr(ws.Cells(r, 2).Value))
        Dim cause   As String: cause   = EscapeJson(CStr(ws.Cells(r, 3).Value))
        Dim traite  As String: traite  = EscapeJson(CStr(ws.Cells(r, 4).Value))
        Dim urgence As String: urgence = EscapeJson(CStr(ws.Cells(r, 5).Value))

        Dim body As String
        body = "{" & _
               """description"":""" & desc    & """," & _
               """demandeur"":"""   & demand  & """," & _
               """cause"":"""       & cause   & """," & _
               """traite"":"""      & traite  & """," & _
               """urgence"":"""     & urgence & """" & _
               "}"

        Dim resp As String
        resp = HttpPost(SERVER_URL & "/predict", body)

        If resp <> "" Then
            ws.Cells(r, 6).Value = ExtractPrediction(resp, "omv")
            ws.Cells(r, 7).Value = ExtractPrediction(resp, "service")
            ws.Cells(r, 8).Value = ExtractPrediction(resp, "origine")
        End If

Suivant:
    Next r
    MsgBox "Classification terminee.", vbInformation, "CIRVIE"
End Sub


' ============================================================
'  Helpers HTTP
' ============================================================

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
    ' Chemin de l'exe : meme dossier que le classeur
    Dim exePath As String
    exePath = ThisWorkbook.Path & "\predict_server.exe"

    If Dir(exePath) = "" Then
        MsgBox "predict_server.exe introuvable dans :" & vbCrLf & ThisWorkbook.Path, _
               vbCritical, "CIRVIE"
        Exit Sub
    End If

    Shell """" & exePath & """", vbHide

    ' Attendre que le serveur soit pret (max START_TIMEOUT_S secondes)
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
'  Parseur JSON minimal — extrait "prediction" d'un bloc
'  Ex. resp = {"omv":{"prediction":"OUI","confidence":98.4,...},...}
' ============================================================
Private Function ExtractPrediction(jsonStr As String, bloc As String) As String
    ' Localise le bloc ("omv", "service" ou "origine") puis extrait "prediction"
    Dim startBloc As Long
    startBloc = InStr(1, jsonStr, """" & bloc & """", vbTextCompare)
    If startBloc = 0 Then
        ExtractPrediction = ""
        Exit Function
    End If

    Dim sub1 As String
    sub1 = Mid(jsonStr, startBloc)

    ' Cherche "prediction":"VALEUR"
    Dim re As Object
    Set re = CreateObject("VBScript.RegExp")
    re.Pattern = """prediction""\s*:\s*""([^""]+)"""
    re.IgnoreCase = True

    Dim m As Object
    Set m = re.Execute(sub1)
    If m.Count > 0 Then
        ExtractPrediction = m.Item(0).SubMatches(0)
    Else
        ExtractPrediction = ""
    End If
End Function


' ============================================================
'  Echappe les caracteres speciaux pour JSON
' ============================================================
Private Function EscapeJson(s As String) As String
    s = Replace(s, "\", "\\")
    s = Replace(s, """", "\""")
    s = Replace(s, Chr(8),  "\b")
    s = Replace(s, Chr(9),  "\t")
    s = Replace(s, Chr(10), "\n")
    s = Replace(s, Chr(13), "\r")
    EscapeJson = s
End Function

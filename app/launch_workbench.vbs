Set shell = CreateObject("WScript.Shell")
Set fso = CreateObject("Scripting.FileSystemObject")
scriptDir = fso.GetParentFolderName(WScript.ScriptFullName)
pythonw = fso.BuildPath(scriptDir, ".venv\Scripts\pythonw.exe")
If Not fso.FileExists(pythonw) Then
    MsgBox "Python environment not found. Expected: app\.venv\Scripts\pythonw.exe", 48, "Virtual PBF"
    WScript.Quit 1
End If
shell.CurrentDirectory = scriptDir
srcDir = fso.BuildPath(scriptDir, "src")
shell.Environment("PROCESS")("PYTHONPATH") = srcDir & ";" & shell.Environment("PROCESS")("PYTHONPATH")
shell.Run """" & pythonw & """ """ & fso.BuildPath(scriptDir, "launch_workbench.pyw") & """", 0, False

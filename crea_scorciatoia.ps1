# Crea una scorciatoia "DeepSight" sul Desktop, con l'icona personalizzata,
# puntata a DeepSight.bat. Da eseguire una sola volta (doppio click con
# tasto destro > "Esegui con PowerShell", oppure da terminale PowerShell).

$cartellaProgetto = Split-Path -Parent $MyInvocation.MyCommand.Path
$bat  = Join-Path $cartellaProgetto "DeepSight.bat"
$ico  = Join-Path $cartellaProgetto "assets\deepsight.ico"
$desktop = [Environment]::GetFolderPath("Desktop")
$lnk = Join-Path $desktop "DeepSight.lnk"

if (-not (Test-Path $bat)) {
    Write-Host "Non trovo DeepSight.bat in $cartellaProgetto" -ForegroundColor Red
    exit 1
}

$shell = New-Object -ComObject WScript.Shell
$shortcut = $shell.CreateShortcut($lnk)
$shortcut.TargetPath = $bat
$shortcut.WorkingDirectory = $cartellaProgetto
if (Test-Path $ico) {
    $shortcut.IconLocation = $ico
}
$shortcut.Description = "Avvia ExoVision (DeepSight)"
$shortcut.Save()

Write-Host "Scorciatoia creata sul Desktop: $lnk" -ForegroundColor Green

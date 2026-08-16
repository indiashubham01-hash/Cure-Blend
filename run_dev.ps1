# CureBlend PowerShell Dev Launcher
$env:Path = "C:\Program Files\nodejs;" + $env:Path
Write-Host "========================================================" -ForegroundColor Cyan
Write-Host "  Starting CureBlend (FastAPI Backend + Vite Frontend)" -ForegroundColor Green
Write-Host "========================================================" -ForegroundColor Cyan
& "C:\Program Files\nodejs\npm.cmd" run dev

@echo off
:: Startet SETUP_GITHUB.ps1 mit den richtigen PowerShell-Rechten
:: Doppelklick genuegt

powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0SETUP_GITHUB.ps1"
pause

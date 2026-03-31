# ==============================================================================
# SYNC_GITHUB.ps1
# Automatischer Sync: pusht Aenderungen zu GitHub (nur wenn vorhanden)
# Wird vom Windows Task Scheduler ausgefuehrt (alle 2 Stunden)
# Kann auch manuell per Doppelklick gestartet werden
# ==============================================================================

$SCRIPT_DIR   = Split-Path -Parent $MyInvocation.MyCommand.Path
$LOG_FILE     = Join-Path $SCRIPT_DIR ".git\sync_log.txt"
$TIMESTAMP    = Get-Date -Format "yyyy-MM-dd HH:mm:ss"

function Write-Log($msg) {
    $line = "[$TIMESTAMP] $msg"
    Add-Content -Path $LOG_FILE -Value $line -Encoding UTF8
    Write-Host $line
}

Set-Location $SCRIPT_DIR

# ── Sicherstellen dass git-Repo vorhanden ─────────────────────────────────────
if (-not (Test-Path ".git")) {
    Write-Log "FEHLER: Kein git-Repository gefunden. Bitte zuerst SETUP_GITHUB.ps1 ausfuehren."
    exit 1
}

# ── Remote auf Erreichbarkeit pruefen ─────────────────────────────────────────
try {
    git ls-remote --exit-code origin HEAD 2>$null | Out-Null
} catch {
    Write-Log "WARNUNG: GitHub nicht erreichbar (kein Internet?). Sync wird uebersprungen."
    exit 0
}

# ── Neue Commits vom Remote holen (pull) ──────────────────────────────────────
Write-Log "Pull von GitHub ..."
git fetch origin main 2>&1 | Out-Null

$localHash  = git rev-parse HEAD 2>$null
$remoteHash = git rev-parse origin/main 2>$null

if ($remoteHash -and ($localHash -ne $remoteHash)) {
    Write-Log "Remote hat neue Commits, merge ..."
    git merge --ff-only origin/main 2>&1 | Out-Null
    Write-Log "Pull abgeschlossen."
}

# ── Lokale Aenderungen pruefen und pushen ─────────────────────────────────────
$changes = git status --porcelain 2>$null

if (-not $changes) {
    Write-Log "Keine Aenderungen. Sync nicht noetig."
    exit 0
}

Write-Log "Aenderungen gefunden, committen und pushen ..."

# Alle Aenderungen stagen (ausser .gitignore-Eintraege)
git add --all

# Geaenderte Dateien fuer Commit-Nachricht zusammenstellen
$changedFiles = (git diff --cached --name-only) -join ", "
$commitMsg    = "Auto-sync $TIMESTAMP

Geaenderte Dateien: $changedFiles"

git commit -m $commitMsg 2>&1 | Out-Null

# Push
$pushResult = git push origin main 2>&1
if ($LASTEXITCODE -eq 0) {
    Write-Log "Push erfolgreich: $changedFiles"
} else {
    Write-Log "FEHLER beim Push: $pushResult"
    exit 1
}

exit 0

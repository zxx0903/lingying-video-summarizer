$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
$py = Join-Path $root 'runtime\python311\python.exe'
if (-not (Test-Path $py)) { Write-Error "Python not found: $py"; exit 1 }

$env:PYTHONUTF8 = '1'
$env:PYTHONPATH = $root

Write-Host 'Starting LingYing demo at http://127.0.0.1:8501'
$ErrorActionPreference = 'Continue'
& $py -m streamlit run (Join-Path $root 'lingying\app.py') `
    --server.address 127.0.0.1 `
    --server.port 8501 `
    --server.headless true `
    --browser.gatherUsageStats false
exit $LASTEXITCODE

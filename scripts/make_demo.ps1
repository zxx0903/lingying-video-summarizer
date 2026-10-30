$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
$py = Join-Path $root 'runtime\python311\python.exe'
if (-not (Test-Path $py)) { Write-Error "Python not found: $py"; exit 1 }

$env:PYTHONUTF8 = '1'
$env:PYTHONPATH = $root
$ErrorActionPreference = 'Continue'
& $py (Join-Path $root 'scripts\make_demo.py')
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

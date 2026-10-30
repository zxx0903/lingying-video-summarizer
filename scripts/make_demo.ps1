$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
. (Join-Path $PSScriptRoot 'python_env.ps1')
$py = Resolve-ProjectPython -Root $root

$env:PYTHONUTF8 = '1'
$env:PYTHONPATH = $root
$ErrorActionPreference = 'Continue'
& $py (Join-Path $root 'scripts\make_demo.py')
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

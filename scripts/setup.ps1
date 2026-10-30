$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
$py = Join-Path $root 'runtime\python311\python.exe'

if (-not (Test-Path $py)) {
    Write-Error "Portable Python not found at $py. Re-run the runtime setup first."
    exit 1
}

$env:PIP_CACHE_DIR = Join-Path $root 'runtime\pip-cache'
$env:TMP = Join-Path $root 'runtime\tmp'
$env:TEMP = $env:TMP
New-Item -ItemType Directory -Force -Path $env:PIP_CACHE_DIR, $env:TMP | Out-Null

Write-Host 'Installing core dependencies...'
$ErrorActionPreference = 'Continue'
& $py -m pip install --no-warn-script-location --progress-bar off `
    -i https://pypi.tuna.tsinghua.edu.cn/simple `
    -r (Join-Path $root 'requirements-core.txt')
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host ''
Write-Host 'Core dependencies installed. Optional ASR/TTS:'
Write-Host "  & $py -m pip install -r $root\requirements-optional.txt"
Write-Host ''
Write-Host 'Next: run scripts\make_demo.ps1, then scripts\run_demo.ps1'

param(
    [switch]$Optional
)

$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
$venv = Join-Path $root '.venv'
$venvPython = Join-Path $venv 'Scripts\python.exe'

function Resolve-BasePython {
    $launcher = Get-Command py -ErrorAction SilentlyContinue
    if ($launcher) {
        foreach ($version in @('3.11', '3.10')) {
            & $launcher.Source "-$version" -c "import sys; raise SystemExit(0 if sys.version_info >= (3, 10) else 1)" 2>$null
            if ($LASTEXITCODE -eq 0) {
                return @{ Exe = $launcher.Source; Prefix = @("-$version") }
            }
        }
    }

    $python = Get-Command python -ErrorAction SilentlyContinue
    if ($python) {
        & $python.Source -c "import sys; raise SystemExit(0 if sys.version_info >= (3, 10) else 1)" 2>$null
        if ($LASTEXITCODE -eq 0) {
            return @{ Exe = $python.Source; Prefix = @() }
        }
    }
    throw 'Python 3.10 or 3.11 was not found. Install Python 3.11, then rerun setup.ps1.'
}

if (-not (Test-Path -LiteralPath $venvPython)) {
    $base = Resolve-BasePython
    Write-Host 'Creating .venv ...'
    & $base.Exe @($base.Prefix) -m venv $venv
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}

$cacheRoot = Join-Path $root 'runtime'
$env:PIP_CACHE_DIR = Join-Path $cacheRoot 'pip-cache'
$env:TMP = Join-Path $cacheRoot 'tmp'
$env:TEMP = $env:TMP
New-Item -ItemType Directory -Force -Path $env:PIP_CACHE_DIR, $env:TMP | Out-Null

$requirements = if ($Optional) { 'requirements-optional.txt' } else { 'requirements-core.txt' }
Write-Host "Installing $requirements ..."
$ErrorActionPreference = 'Continue'
& $venvPython -m pip install --no-warn-script-location --progress-bar off `
    -r (Join-Path $root $requirements)
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host ''
Write-Host 'Environment ready.'
Write-Host 'Run: powershell -ExecutionPolicy Bypass -File .\scripts\run_demo.ps1'
if (-not $Optional) {
    Write-Host 'For speech transcription and OCR: .\scripts\setup.ps1 -Optional'
}

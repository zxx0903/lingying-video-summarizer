$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
. (Join-Path $PSScriptRoot 'python_env.ps1')
$py = Resolve-ProjectPython -Root $root
$video = Join-Path $root 'data\input\demo_lecture.mp4'
$out = Join-Path $root 'outputs\smoke_test'

if (-not (Test-Path $video)) {
    Write-Host 'Demo video not found, generating it first...'
    & powershell -NoProfile -ExecutionPolicy Bypass -File (Join-Path $root 'scripts\make_demo.ps1')
}

$env:PYTHONUTF8 = '1'
$env:PYTHONPATH = $root
$ErrorActionPreference = 'Continue'
& $py -m lingying.cli --input $video --out $out --target 90 --no-asr
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

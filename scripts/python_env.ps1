function Resolve-ProjectPython {
    param([Parameter(Mandatory = $true)][string]$Root)

    $candidates = @(
        (Join-Path $Root 'runtime\python311\python.exe'),
        (Join-Path $Root '.venv\Scripts\python.exe')
    )
    foreach ($candidate in $candidates) {
        if (-not (Test-Path -LiteralPath $candidate)) {
            continue
        }
        & $candidate -c "import sys; raise SystemExit(0 if sys.version_info >= (3, 10) else 1)"
        if ($LASTEXITCODE -eq 0) {
            return (Resolve-Path -LiteralPath $candidate).Path
        }
    }

    throw (
        "Project Python 3.10+ was not found. Run .\scripts\setup.ps1 " +
        "from the project root. The first setup requires internet access."
    )
}

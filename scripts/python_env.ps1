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
        "未找到 Python 3.10+ 项目环境。请先在项目根目录运行 " +
        ".\scripts\setup.ps1；首次运行需要联网下载依赖。"
    )
}

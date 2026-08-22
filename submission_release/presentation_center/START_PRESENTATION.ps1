$ErrorActionPreference = "Stop"
$center = Split-Path -Parent $MyInvocation.MyCommand.Path
$root = (Resolve-Path (Join-Path $center "..\..")).Path
$python = Join-Path $root ".venv\Scripts\python.exe"

if (-not (Test-Path -LiteralPath $python)) {
    throw "Project virtual environment not found: $python"
}

Set-Location -LiteralPath $root
& $python -m vessel_tree_generator present

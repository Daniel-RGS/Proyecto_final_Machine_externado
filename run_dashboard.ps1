$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$runtimePython = Join-Path $env:USERPROFILE ".cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe"

if (Test-Path $runtimePython) {
    $python = $runtimePython
} else {
    $python = "python"
}

Push-Location $projectRoot
try {
    & $python -m streamlit run dashboard\app.py
} finally {
    Pop-Location
}

[CmdletBinding()]
param(
    [switch]$BuildTools
)

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$uvCommand = Get-Command uv -ErrorAction SilentlyContinue
if ($null -eq $uvCommand) {
    throw 'uv was not found on PATH. Install uv, reopen PowerShell, and run this script again.'
}

Push-Location $repoRoot
try {
    $arguments = @('sync', '--frozen')
    if ($BuildTools) {
        $arguments += @('--extra', 'build')
    }
    & $uvCommand.Source @arguments
    if ($LASTEXITCODE -ne 0) {
        throw "uv sync failed with exit code $LASTEXITCODE"
    }
    & $uvCommand.Source run --frozen python -c "import sys; assert sys.version_info[:2] == (3, 12); print(sys.version)"
    if ($LASTEXITCODE -ne 0) {
        throw 'The project environment is not running Python 3.12.'
    }
}
finally {
    Pop-Location
}

Write-Host "LoRA Factory development environment is ready at $repoRoot\.venv"

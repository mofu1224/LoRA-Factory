[CmdletBinding()]
param(
    [switch]$Console
)

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$pythonWindowed = Join-Path $repoRoot '.venv\Scripts\pythonw.exe'
$uvCommand = Get-Command uv -ErrorAction SilentlyContinue

if ($Console) {
    if ($null -eq $uvCommand) {
        throw 'uv was not found on PATH.'
    }
    Push-Location $repoRoot
    try {
        & $uvCommand.Source run --frozen lora-factory-gui
        exit $LASTEXITCODE
    }
    finally {
        Pop-Location
    }
}

if (-not (Test-Path -LiteralPath $pythonWindowed -PathType Leaf)) {
    throw 'The project environment is missing. Run scripts\bootstrap.ps1 first.'
}

$process = Start-Process `
    -FilePath $pythonWindowed `
    -ArgumentList @('-m', 'lora_factory') `
    -WorkingDirectory $repoRoot `
    -WindowStyle Hidden `
    -PassThru
Write-Host "LoRA Factory GUI started without a console window (PID $($process.Id))."

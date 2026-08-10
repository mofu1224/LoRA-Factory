[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string]$BaseModel,

    [Parameter(Mandatory = $true)]
    [string]$InputPath,

    [Parameter(Mandatory = $true)]
    [string]$OutputRoot,

    [string[]]$GpuUuid,
    [string]$Trigger = 'lfx_smoke',
    [string]$RuntimeRoot,
    [switch]$InstallRuntime,
    [switch]$RequireCodex
)

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$uvCommand = Get-Command uv -ErrorAction SilentlyContinue
if ($null -eq $uvCommand) {
    throw 'uv was not found on PATH.'
}

Push-Location $repoRoot
try {
    if ($InstallRuntime) {
        $installArguments = @('run', '--frozen', 'lora-factory', 'install-runtime')
        if ($RuntimeRoot) {
            $installArguments += @('--runtime-root', $RuntimeRoot)
        }
        & $uvCommand.Source @installArguments
        if ($LASTEXITCODE -ne 0) {
            throw "Managed runtime installation failed with exit code $LASTEXITCODE"
        }
    }

    $arguments = @(
        'run', '--frozen', 'lora-factory', 'live-smoke',
        '--base-model', (Resolve-Path -LiteralPath $BaseModel).Path,
        '--input', (Resolve-Path -LiteralPath $InputPath).Path,
        '--output-root', $OutputRoot,
        '--trigger', $Trigger
    )
    foreach ($uuid in $GpuUuid) {
        $arguments += @('--gpu-uuid', $uuid)
    }
    if ($RuntimeRoot) {
        $arguments += @('--runtime-root', $RuntimeRoot)
    }
    if ($RequireCodex) {
        $arguments += '--require-codex'
    }
    & $uvCommand.Source @arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Real backend smoke failed with exit code $LASTEXITCODE"
    }
}
finally {
    Pop-Location
}

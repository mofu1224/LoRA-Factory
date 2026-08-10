[CmdletBinding()]
param(
    [switch]$SkipFakeE2E
)

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$uvCommand = Get-Command uv -ErrorAction SilentlyContinue
if ($null -eq $uvCommand) {
    throw 'uv was not found on PATH.'
}

$env:UV_CACHE_DIR = Join-Path $repoRoot '.uv-cache\publication'
Push-Location $repoRoot
try {
    & git diff --check
    if ($LASTEXITCODE -ne 0) { throw 'git diff --check failed.' }
    & $uvCommand.Source lock --check
    if ($LASTEXITCODE -ne 0) { throw 'uv lock --check failed.' }
    & $uvCommand.Source sync --frozen
    if ($LASTEXITCODE -ne 0) { throw 'uv sync --frozen failed.' }
    & $uvCommand.Source run ruff format --check .
    if ($LASTEXITCODE -ne 0) { throw 'Ruff format check failed.' }
    & $uvCommand.Source run ruff check .
    if ($LASTEXITCODE -ne 0) { throw 'Ruff lint failed.' }
    & $uvCommand.Source run mypy src
    if ($LASTEXITCODE -ne 0) { throw 'mypy failed.' }
    & $uvCommand.Source run pytest `
        --basetemp .test-tmp\publication `
        --cov=lora_factory `
        --cov-branch `
        --cov-report=term
    if ($LASTEXITCODE -ne 0) { throw 'Test or coverage gate failed.' }
    if (-not $SkipFakeE2E) {
        & $uvCommand.Source run --frozen lora-factory fake-e2e `
            --preset character `
            --image-count 8 `
            --json
        if ($LASTEXITCODE -ne 0) { throw 'Fake E2E failed.' }
    }
}
finally {
    Pop-Location
}

Write-Host 'Publication preflight passed. Review the staged diff before any push.'

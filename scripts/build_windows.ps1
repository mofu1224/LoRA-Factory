[CmdletBinding()]
param(
    [string]$BuildLabel = (Get-Date -Format 'yyyyMMdd-HHmmss'),
    [switch]$SystemVcRuntime
)

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest

if ($BuildLabel -notmatch '^[0-9A-Za-z._-]+$') {
    throw 'BuildLabel may contain only letters, digits, dot, underscore, and hyphen.'
}

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$uvCommand = Get-Command uv -ErrorAction SilentlyContinue
if ($null -eq $uvCommand) {
    throw 'uv was not found on PATH.'
}
if ([string]::IsNullOrWhiteSpace($env:UV_CACHE_DIR)) {
    $env:UV_CACHE_DIR = Join-Path $repoRoot '.uv-cache-build'
}
$distRoot = Join-Path $repoRoot "dist\windows-$BuildLabel"
$workRoot = Join-Path $repoRoot "build\pyinstaller-$BuildLabel"
if (Test-Path -LiteralPath $distRoot) {
    throw "Build destination already exists; refusing overwrite: $distRoot"
}
if (Test-Path -LiteralPath $workRoot) {
    throw "Build work directory already exists; refusing overwrite: $workRoot"
}

Push-Location $repoRoot
try {
    & $uvCommand.Source sync --frozen --extra build
    if ($LASTEXITCODE -ne 0) {
        throw "uv sync failed with exit code $LASTEXITCODE"
    }
    & $uvCommand.Source run --frozen --extra build pyinstaller `
        --noconfirm `
        --clean `
        --distpath $distRoot `
        --workpath $workRoot `
        packaging\lora_factory.spec
    if ($LASTEXITCODE -ne 0) {
        throw "PyInstaller failed with exit code $LASTEXITCODE"
    }
}
finally {
    Pop-Location
}

$appDirectory = Join-Path $distRoot 'LoRA Factory'
$executable = Join-Path $appDirectory 'LoRA Factory.exe'
if (-not (Test-Path -LiteralPath $executable -PathType Leaf)) {
    throw "Build completed without the expected executable: $executable"
}
Copy-Item -LiteralPath (Join-Path $repoRoot 'README.md') -Destination $appDirectory
Copy-Item -LiteralPath (Join-Path $repoRoot 'LICENSE') -Destination $appDirectory
Copy-Item -LiteralPath (Join-Path $repoRoot 'THIRD_PARTY_NOTICES.md') -Destination $appDirectory
Copy-Item -LiteralPath (Join-Path $repoRoot 'scripts\install_vcredist.ps1') -Destination $appDirectory
Copy-Item -LiteralPath (Join-Path $repoRoot 'scripts\install_vcredist.cmd') -Destination $appDirectory
$licenseDirectory = Join-Path $appDirectory 'licenses'
& $uvCommand.Source run --frozen --extra build python -m lora_factory.packaging.license_export `
    --output $licenseDirectory
if ($LASTEXITCODE -ne 0) {
    throw "Third-party license export failed with exit code $LASTEXITCODE"
}
$pythonLicense = Get-ChildItem -LiteralPath $licenseDirectory -Recurse -File -Filter 'LICENSE.txt' |
    Where-Object { $_.Directory.Name -like 'CPython-*' } |
    Select-Object -First 1
if ($null -eq $pythonLicense) {
    throw 'The packaged license bundle does not contain the CPython license.'
}
$qtMaterial = Join-Path $repoRoot 'licenses\Qt'
$msvcMaterial = Join-Path $repoRoot 'licenses\Microsoft-Visual-Cpp'
foreach ($requiredMaterial in @($qtMaterial, $msvcMaterial)) {
    if (-not (Test-Path -LiteralPath $requiredMaterial -PathType Container)) {
        throw "Required distribution license material is missing: $requiredMaterial"
    }
}
Copy-Item -LiteralPath $qtMaterial -Destination (Join-Path $licenseDirectory 'Qt') -Recurse
Copy-Item -LiteralPath $msvcMaterial -Destination (Join-Path $licenseDirectory 'Microsoft-Visual-Cpp') -Recurse
$systemVcRuntimeNames = @(
    'VCRUNTIME140.dll'
    'VCRUNTIME140_1.dll'
    'MSVCP140.dll'
    'MSVCP140_1.dll'
    'MSVCP140_2.dll'
    'CONCRT140.dll'
    'VCOMP140.dll'
)
$msvcRuntimeFiles = @(Get-ChildItem -LiteralPath $appDirectory -Recurse -File |
    Where-Object { $_.Name -in $systemVcRuntimeNames })
if ($SystemVcRuntime) {
    foreach ($runtimeFile in $msvcRuntimeFiles) {
        Remove-Item -LiteralPath $runtimeFile.FullName -Force
    }
    $remainingMsvcFiles = @(Get-ChildItem -LiteralPath $appDirectory -Recurse -File |
        Where-Object { $_.Name -in $systemVcRuntimeNames })
    if ($remainingMsvcFiles.Count -ne 0) {
        throw 'SystemVcRuntime mode still contains Microsoft runtime DLLs.'
    }
    Write-Host 'SystemVcRuntime mode: standard system VC++ runtime DLLs are omitted; install the official x64 VC++ Redistributable before launch.'
}
$lgplText = Join-Path $licenseDirectory 'Qt\LGPL-3.0-only.txt'
$gplText = Join-Path $licenseDirectory 'Qt\GPL-3.0-only.txt'
if (-not (Test-Path -LiteralPath $lgplText -PathType Leaf) -or
    -not (Test-Path -LiteralPath $gplText -PathType Leaf)) {
    throw 'The packaged Qt license bundle is incomplete.'
}
$digest = (Get-FileHash -LiteralPath $executable -Algorithm SHA256).Hash
Write-Host "Windows build: $appDirectory"
Write-Host "Executable SHA256: $digest"
Write-Host 'The separately installed managed CUDA runtime is intentionally not embedded.'

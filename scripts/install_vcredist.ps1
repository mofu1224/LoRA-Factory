[CmdletBinding()]
param()

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest

$packageId = 'Microsoft.VCRedist.2015+.x64'
$officialDownload = 'https://learn.microsoft.com/en-us/cpp/windows/latest-supported-vc-redist?view=msvc-170'
$runtimeFiles = @(
    Join-Path $env:WINDIR 'System32\VCRUNTIME140.dll'
    Join-Path $env:WINDIR 'System32\VCRUNTIME140_1.dll'
    Join-Path $env:WINDIR 'System32\MSVCP140.dll'
    Join-Path $env:WINDIR 'System32\MSVCP140_1.dll'
    Join-Path $env:WINDIR 'System32\MSVCP140_2.dll'
)

if (@($runtimeFiles | Where-Object { -not (Test-Path -LiteralPath $_ -PathType Leaf) }).Count -eq 0) {
    Write-Host 'Microsoft Visual C++ x64 Redistributable is already installed.'
    exit 0
}

$winget = Get-Command winget.exe -ErrorAction SilentlyContinue
if ($null -eq $winget) {
    Write-Warning 'WinGet is not available on this Windows installation.'
    Write-Host 'Open the official Microsoft download page instead:'
    Start-Process $officialDownload
    exit 2
}

Write-Host "Installing $packageId from the official WinGet source."
Write-Host 'The Microsoft package and source agreements will be accepted for this installation.'
$arguments = @(
    'install'
    '--id'
    $packageId
    '--exact'
    '--source'
    'winget'
    '--accept-package-agreements'
    '--accept-source-agreements'
    '--silent'
)
& $winget.Source @arguments
if ($LASTEXITCODE -ne 0) {
    throw "WinGet failed to install $packageId with exit code $LASTEXITCODE."
}

$missing = @($runtimeFiles | Where-Object { -not (Test-Path -LiteralPath $_ -PathType Leaf) })
if ($missing.Count -ne 0) {
    throw "The VC++ Redistributable installer completed, but required files are still missing: $($missing -join ', ')"
}

Write-Host 'Microsoft Visual C++ x64 Redistributable is ready. You can now launch LoRA Factory.exe.'

[CmdletBinding()]
param()

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest

$packageId = 'Microsoft.VCRedist.2015+.x64'
$directInstallerUrl = 'https://aka.ms/vc14/vc_redist.x64.exe'
$officialDownload = 'https://learn.microsoft.com/en-us/cpp/windows/latest-supported-vc-redist?view=msvc-170'
$runtimeFiles = @(
    Join-Path $env:WINDIR 'System32\VCRUNTIME140.dll'
    Join-Path $env:WINDIR 'System32\VCRUNTIME140_1.dll'
    Join-Path $env:WINDIR 'System32\MSVCP140.dll'
    Join-Path $env:WINDIR 'System32\MSVCP140_1.dll'
    Join-Path $env:WINDIR 'System32\MSVCP140_2.dll'
)

function Get-MissingRuntimeFiles {
    return @($runtimeFiles | Where-Object { -not (Test-Path -LiteralPath $_ -PathType Leaf) })
}

function Install-DirectVcRedist {
    $installerPath = Join-Path ([System.IO.Path]::GetTempPath()) (
        'LoRAFactory-vc_redist.x64-{0}.exe' -f [Guid]::NewGuid().ToString('N')
    )

    try {
        Write-Host 'WinGet is unavailable or could not install the package.'
        Write-Host 'Downloading the official Microsoft Visual C++ x64 Redistributable.'
        Invoke-WebRequest -Uri $directInstallerUrl -OutFile $installerPath -UseBasicParsing

        if (-not (Test-Path -LiteralPath $installerPath -PathType Leaf)) {
            throw 'The Microsoft installer was not downloaded.'
        }

        $signature = Get-AuthenticodeSignature -FilePath $installerPath
        $signerName = if ($null -ne $signature.SignerCertificate) {
            $signature.SignerCertificate.GetNameInfo(
                [System.Security.Cryptography.X509Certificates.X509NameType]::SimpleName,
                $false
            )
        } else {
            ''
        }
        if ($signature.Status -ne 'Valid' -or $signerName -ne 'Microsoft Corporation') {
            throw 'The downloaded installer did not pass the Microsoft Authenticode signature check.'
        }

        Write-Host 'Starting the Microsoft installer. Windows may ask for administrator approval.'
        $process = Start-Process `
            -FilePath $installerPath `
            -ArgumentList @('/install', '/quiet', '/norestart') `
            -Verb RunAs `
            -Wait `
            -PassThru
        if ($process.ExitCode -notin @(0, 1638, 3010)) {
            throw "The Microsoft VC++ installer returned exit code $($process.ExitCode)."
        }
    } catch {
        throw "The official Microsoft VC++ installation failed: $($_.Exception.Message) Open $officialDownload for manual installation."
    } finally {
        if (Test-Path -LiteralPath $installerPath -PathType Leaf) {
            Remove-Item -LiteralPath $installerPath -Force -ErrorAction SilentlyContinue
        }
    }
}

$missing = @(Get-MissingRuntimeFiles)
if ($missing.Count -eq 0) {
    Write-Host 'Microsoft Visual C++ x64 Redistributable is already installed.'
    exit 0
}

$winget = Get-Command winget.exe -ErrorAction SilentlyContinue
if ($null -eq $winget) {
    Write-Warning 'WinGet is not available on this Windows installation.'
    Install-DirectVcRedist
} else {
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
        Write-Warning "WinGet failed to install $packageId with exit code $LASTEXITCODE."
        Install-DirectVcRedist
    }
}

$missing = @(Get-MissingRuntimeFiles)
if ($missing.Count -ne 0) {
    throw "The VC++ Redistributable installer completed, but required files are still missing: $($missing -join ', ')"
}

Write-Host 'Microsoft Visual C++ x64 Redistributable is ready. You can now launch LoRA Factory.exe.'

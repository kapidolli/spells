param(
    [Parameter(Mandatory = $true)][string]$Path
)

$ErrorActionPreference = 'Stop'

$thumbprint = $env:SPELLS_SIGN_THUMBPRINT
if (-not $thumbprint) {
    throw 'SPELLS_SIGN_THUMBPRINT is not set: give the thumbprint of the code signing certificate in Cert:\CurrentUser\My'
}
$timestamp = $env:SPELLS_SIGN_TIMESTAMP
if (-not $timestamp) {
    $timestamp = 'http://time.certum.pl'
}

$certificate = Get-ChildItem -Path "Cert:\CurrentUser\My\$thumbprint" -CodeSigningCert
if (-not $certificate) {
    throw "No code signing certificate with thumbprint $thumbprint in Cert:\CurrentUser\My"
}

$result = Set-AuthenticodeSignature -FilePath $Path -Certificate $certificate -HashAlgorithm SHA256 -TimestampServer $timestamp
if ($result.Status -eq 'Valid') {
    exit 0
}
if ($env:SPELLS_SIGN_ALLOW_UNTRUSTED -eq '1' -and $result.SignerCertificate -and $result.SignerCertificate.Thumbprint -eq $certificate.Thumbprint) {
    Write-Host "signed $Path with an untrusted test certificate ($($result.Status))"
    exit 0
}
throw "Signing $Path failed: $($result.Status) $($result.StatusMessage)"

[CmdletBinding()]
param(
    [string]$RepoPath = "",
    [string]$BaseBranch = "integration"
)

$ErrorActionPreference = "Stop"

function Test-IsAllowedPath {
    param(
        [string]$FilePath,
        [string[]]$AllowedRules
    )

    $normalizedFile = $FilePath.Replace("\", "/")
    foreach ($rule in $AllowedRules) {
        $normalizedRule = $rule.Replace("\", "/")
        if ($normalizedRule.EndsWith("/")) {
            if ($normalizedFile.StartsWith($normalizedRule, [System.StringComparison]::OrdinalIgnoreCase)) {
                return $true
            }
        }
        else {
            if ($normalizedFile.Equals($normalizedRule, [System.StringComparison]::OrdinalIgnoreCase)) {
                return $true
            }
        }
    }
    return $false
}

if ([string]::IsNullOrWhiteSpace($RepoPath)) {
    $scriptDir = Split-Path -Parent $PSCommandPath
    $RepoPath = (Resolve-Path (Join-Path $scriptDir "..")).Path
}
else {
    $RepoPath = (Resolve-Path $RepoPath).Path
}
$ownerFile = Join-Path $RepoPath ".coordination/ownership.json"

if (-not (Test-Path $ownerFile)) {
    $scriptDir = Split-Path -Parent $PSCommandPath
    $fallbackRepo = (Resolve-Path (Join-Path $scriptDir "..")).Path
    $fallbackOwnerFile = Join-Path $fallbackRepo ".coordination/ownership.json"

    if (Test-Path $fallbackOwnerFile) {
        $ownerFile = $fallbackOwnerFile
    }
    else {
        throw "Nu exista $ownerFile."
    }
}

$branch = (& git -C $RepoPath branch --show-current).Trim()
if ($LASTEXITCODE -ne 0) {
    throw "Nu pot citi branch-ul curent."
}

& git -C $RepoPath fetch origin | Out-Null
if ($LASTEXITCODE -ne 0) {
    throw "Nu pot face fetch origin."
}

$changed = & git -C $RepoPath diff --name-only "origin/$BaseBranch...HEAD"
if ($LASTEXITCODE -ne 0) {
    throw "Nu pot obtine lista de fisiere schimbate fata de origin/$BaseBranch."
}
$changedFiles = @($changed | Where-Object { -not [string]::IsNullOrWhiteSpace($_) })

if ($changedFiles.Count -eq 0) {
    Write-Host "Ownership check: fara fisiere modificate fata de origin/$BaseBranch."
    exit 0
}

$ownership = Get-Content -Raw -Path $ownerFile | ConvertFrom-Json
$defaultShared = @($ownership.defaultShared)
$branchProperty = $ownership.owners.PSObject.Properties | Where-Object { $_.Name -eq $branch } | Select-Object -First 1

if ($null -eq $branchProperty) {
    Write-Warning "Branch '$branch' nu are ownership dedicat. Ruleaza doar cu reguli shared."
    $allowed = $defaultShared
}
else {
    $allowed = $defaultShared + @($branchProperty.Value)
}

$violations = @()
foreach ($file in $changedFiles) {
    if (-not (Test-IsAllowedPath -FilePath $file -AllowedRules $allowed)) {
        $violations += $file
    }
}

if ($violations.Count -gt 0) {
    Write-Host ""
    Write-Host "Ownership check FAILED pentru branch '$branch'."
    Write-Host "Fisiere in afara zonei permise:"
    $violations | ForEach-Object { Write-Host "- $_" }
    exit 1
}

Write-Host ""
Write-Host "Ownership check PASSED pentru branch '$branch'."
Write-Host "Fisiere schimbate:"
$changedFiles | ForEach-Object { Write-Host "- $_" }

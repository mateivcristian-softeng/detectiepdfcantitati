[CmdletBinding()]
param(
    [string]$RepoPath = "",
    [string]$IntegrationBranch = "integration",
    [string[]]$FeatureBranches = @(
        "feat/ai1-quickwins-cv",
        "feat/ai2-medium-classic-ml",
        "feat/ai3-major-foundation",
        "feat/ai4-data-annotations",
        "feat/ai5-metrics-eval"
    ),
    [string]$TestCommand = "python -m compileall .",
    [switch]$NoPush
)

$ErrorActionPreference = "Stop"

function Invoke-Git {
    param(
        [string]$Path,
        [string[]]$GitArgs
    )

    $output = & git -C $Path @GitArgs
    if ($LASTEXITCODE -ne 0) {
        throw "Comanda git a esuat in '$Path': git $($GitArgs -join ' ')"
    }

    return $output
}

if ([string]::IsNullOrWhiteSpace($RepoPath)) {
    $scriptDir = Split-Path -Parent $PSCommandPath
    $RepoPath = (Resolve-Path (Join-Path $scriptDir "..")).Path
}
else {
    $RepoPath = (Resolve-Path $RepoPath).Path
}

$dirty = & git -C $RepoPath status --porcelain
if ($LASTEXITCODE -ne 0) {
    throw "Nu pot verifica statusul repository-ului."
}
if (-not [string]::IsNullOrWhiteSpace(($dirty -join ""))) {
    throw "Repository-ul principal are modificari locale. Commit/stash inainte de merge train."
}

Invoke-Git -Path $RepoPath -GitArgs @("fetch", "origin")

$remoteIntegration = Invoke-Git -Path $RepoPath -GitArgs @("ls-remote", "--heads", "origin", $IntegrationBranch)
if ([string]::IsNullOrWhiteSpace(($remoteIntegration -join ""))) {
    Invoke-Git -Path $RepoPath -GitArgs @("branch", $IntegrationBranch, "main")
    Invoke-Git -Path $RepoPath -GitArgs @("push", "-u", "origin", $IntegrationBranch)
}

Invoke-Git -Path $RepoPath -GitArgs @("switch", $IntegrationBranch)
Invoke-Git -Path $RepoPath -GitArgs @("pull", "--ff-only", "origin", $IntegrationBranch)

foreach ($branch in $FeatureBranches) {
    $remoteBranch = Invoke-Git -Path $RepoPath -GitArgs @("ls-remote", "--heads", "origin", $branch)
    if ([string]::IsNullOrWhiteSpace(($remoteBranch -join ""))) {
        throw "Lipseste remote branch-ul origin/$branch. Opreste integrarea."
    }

    try {
        Invoke-Git -Path $RepoPath -GitArgs @("merge", "--no-ff", "--no-edit", "origin/$branch") | Out-Null
    }
    catch {
        throw "Conflict la merge pentru '$branch'. Rezolva conflictul manual si ruleaza din nou."
    }
}

Push-Location $RepoPath
try {
    Invoke-Expression $TestCommand
    if ($LASTEXITCODE -ne 0) {
        throw "Test command a esuat: $TestCommand"
    }
}
finally {
    Pop-Location
}

if (-not $NoPush) {
    Invoke-Git -Path $RepoPath -GitArgs @("push", "origin", $IntegrationBranch)
}

Write-Host ""
Write-Host "Merge train complet pe '$IntegrationBranch'."
Write-Host "Test command: $TestCommand"
if ($NoPush) {
    Write-Host "NoPush activ: branch-ul NU a fost publicat."
}

[CmdletBinding()]
param(
    [string]$RepoPath = "",
    [string]$WorktreeRoot = "",
    [string]$BaseBranch = "integration",
    [switch]$ForceRecreate
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

function Get-RegisteredWorktreePaths {
    param([string]$Path)

    $lines = Invoke-Git -Path $Path -GitArgs @("worktree", "list", "--porcelain")
    $paths = @()
    foreach ($line in $lines) {
        if ($line -like "worktree *") {
            $paths += $line.Substring(9)
        }
    }
    return $paths
}

if ([string]::IsNullOrWhiteSpace($RepoPath)) {
    $scriptDir = Split-Path -Parent $PSCommandPath
    $RepoPath = (Resolve-Path (Join-Path $scriptDir "..")).Path
}
else {
    $RepoPath = (Resolve-Path $RepoPath).Path
}

if ([string]::IsNullOrWhiteSpace($WorktreeRoot)) {
    $projectName = Split-Path $RepoPath -Leaf
    $parentDir = Split-Path $RepoPath -Parent
    $WorktreeRoot = Join-Path $parentDir "$projectName-WT"
}

New-Item -ItemType Directory -Force -Path $WorktreeRoot | Out-Null

$agents = @(
    [PSCustomObject]@{ Name = "AI1"; Branch = "feat/ai1-quickwins-cv";      Folder = "ai1-quickwins-cv";      Focus = "Quick wins OpenCV (roof/window/facade)" },
    [PSCustomObject]@{ Name = "AI2"; Branch = "feat/ai2-medium-classic-ml"; Folder = "ai2-medium-classic-ml"; Focus = "Refinare clasica + validator" },
    [PSCustomObject]@{ Name = "AI3"; Branch = "feat/ai3-major-foundation";  Folder = "ai3-major-foundation";  Focus = "Adaptor modele segmentare (opt-in)" },
    [PSCustomObject]@{ Name = "AI4"; Branch = "feat/ai4-data-annotations";  Folder = "ai4-data-annotations";  Focus = "Schema/anotari/date" },
    [PSCustomObject]@{ Name = "AI5"; Branch = "feat/ai5-metrics-eval";      Folder = "ai5-metrics-eval";      Focus = "Metrici, evaluator, CI gate" }
)

Invoke-Git -Path $RepoPath -GitArgs @("rev-parse", "--is-inside-work-tree") | Out-Null
Invoke-Git -Path $RepoPath -GitArgs @("fetch", "origin")
Invoke-Git -Path $RepoPath -GitArgs @("switch", "main")
Invoke-Git -Path $RepoPath -GitArgs @("pull", "--ff-only", "origin", "main")

$remoteBaseBranch = Invoke-Git -Path $RepoPath -GitArgs @("ls-remote", "--heads", "origin", $BaseBranch)
$localBaseBranch = Invoke-Git -Path $RepoPath -GitArgs @("branch", "--list", $BaseBranch)

if ([string]::IsNullOrWhiteSpace(($localBaseBranch -join ""))) {
    if ([string]::IsNullOrWhiteSpace(($remoteBaseBranch -join ""))) {
        Invoke-Git -Path $RepoPath -GitArgs @("branch", $BaseBranch, "main")
        Invoke-Git -Path $RepoPath -GitArgs @("push", "-u", "origin", $BaseBranch)
    }
    else {
        Invoke-Git -Path $RepoPath -GitArgs @("switch", "--create", $BaseBranch, "--track", "origin/$BaseBranch")
        Invoke-Git -Path $RepoPath -GitArgs @("switch", "main")
    }
}
elseif (-not [string]::IsNullOrWhiteSpace(($remoteBaseBranch -join ""))) {
    Invoke-Git -Path $RepoPath -GitArgs @("fetch", "origin", "$($BaseBranch):$BaseBranch")
}

$registeredWorktrees = Get-RegisteredWorktreePaths -Path $RepoPath
$result = @()

foreach ($agent in $agents) {
    $targetPath = Join-Path $WorktreeRoot $agent.Folder
    $isRegistered = $registeredWorktrees -contains $targetPath

    if ($ForceRecreate -and $isRegistered) {
        Invoke-Git -Path $RepoPath -GitArgs @("worktree", "remove", "--force", $targetPath)
        $registeredWorktrees = Get-RegisteredWorktreePaths -Path $RepoPath
        $isRegistered = $false
    }

    if ($ForceRecreate -and (Test-Path $targetPath)) {
        Remove-Item -Recurse -Force -Path $targetPath
    }

    if (-not $isRegistered) {
        $branchExists = Invoke-Git -Path $RepoPath -GitArgs @("branch", "--list", $agent.Branch)
        if ([string]::IsNullOrWhiteSpace(($branchExists -join ""))) {
            Invoke-Git -Path $RepoPath -GitArgs @("worktree", "add", $targetPath, "-b", $agent.Branch, $BaseBranch)
        }
        else {
            Invoke-Git -Path $RepoPath -GitArgs @("worktree", "add", $targetPath, $agent.Branch)
        }
    }

    $remoteFeatureBranch = Invoke-Git -Path $RepoPath -GitArgs @("ls-remote", "--heads", "origin", $agent.Branch)
    if ([string]::IsNullOrWhiteSpace(($remoteFeatureBranch -join ""))) {
        Invoke-Git -Path $targetPath -GitArgs @("push", "-u", "origin", $agent.Branch)
    }

    $result += [PSCustomObject]@{
        agent  = $agent.Name
        branch = $agent.Branch
        path   = $targetPath
        focus  = $agent.Focus
    }
}

$coordinationDir = Join-Path $RepoPath ".coordination"
New-Item -ItemType Directory -Force -Path $coordinationDir | Out-Null

$mapFile = Join-Path $coordinationDir "worktree-map.json"
$result | ConvertTo-Json -Depth 3 | Set-Content -Path $mapFile -Encoding UTF8

Write-Host ""
Write-Host "Worktree setup complet:"
$result | Format-Table -AutoSize
Write-Host ""
Write-Host "Map salvat in: $mapFile"

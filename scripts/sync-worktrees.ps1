[CmdletBinding()]
param(
    [string]$RepoPath = "",
    [string]$BaseBranch = "integration",
    [switch]$Push
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
$mapFile = Join-Path $RepoPath ".coordination/worktree-map.json"

if (-not (Test-Path $mapFile)) {
    throw "Nu exista $mapFile. Ruleaza mai intai scripts/setup-worktrees.ps1."
}

$worktrees = Get-Content -Raw -Path $mapFile | ConvertFrom-Json
if ($worktrees -isnot [System.Array]) {
    $worktrees = @($worktrees)
}

Invoke-Git -Path $RepoPath -GitArgs @("fetch", "origin")

$summary = @()
foreach ($entry in $worktrees) {
    $status = "ok"
    $details = ""
    $branch = $entry.branch
    $path = $entry.path

    if (-not (Test-Path $path)) {
        $summary += [PSCustomObject]@{
            agent   = $entry.agent
            branch  = $branch
            status  = "missing"
            details = "worktree path lipsa"
        }
        continue
    }

    try {
        Invoke-Git -Path $path -GitArgs @("fetch", "origin") | Out-Null
        $dirtyLines = & git -C $path status --porcelain
        if ($LASTEXITCODE -ne 0) {
            throw "Nu pot evalua git status."
        }

        if (-not [string]::IsNullOrWhiteSpace(($dirtyLines -join ""))) {
            $status = "skipped"
            $details = "worktree cu modificari locale"
        }
        else {
            Invoke-Git -Path $path -GitArgs @("pull", "--rebase", "origin", $branch) | Out-Null

            if ($branch -ne "main" -and $branch -ne $BaseBranch) {
                Invoke-Git -Path $path -GitArgs @("rebase", "origin/$BaseBranch") | Out-Null
                $details = "rebased on origin/$BaseBranch"
            }
            else {
                $details = "fast-forward from origin/$branch"
            }

            if ($Push) {
                Invoke-Git -Path $path -GitArgs @("push", "origin", $branch) | Out-Null
                $details = "$details + pushed"
            }
        }
    }
    catch {
        $status = "error"
        $details = $_.Exception.Message
    }

    $summary += [PSCustomObject]@{
        agent   = $entry.agent
        branch  = $branch
        status  = $status
        details = $details
    }
}

Write-Host ""
Write-Host "Rezultat sync:"
$summary | Format-Table -AutoSize

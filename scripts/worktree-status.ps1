[CmdletBinding()]
param(
    [string]$RepoPath = ""
)

$ErrorActionPreference = "Stop"

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

$rows = @()
foreach ($entry in $worktrees) {
    $path = $entry.path
    if (-not (Test-Path $path)) {
        $rows += [PSCustomObject]@{
            agent      = $entry.agent
            branch     = $entry.branch
            dirty      = "-"
            ahead      = "-"
            behind     = "-"
            lastCommit = "path lipsa"
        }
        continue
    }

    $branch = (& git -C $path branch --show-current).Trim()
    if ($LASTEXITCODE -ne 0) { throw "Nu pot citi branch in $path" }

    $dirtyLines = & git -C $path status --porcelain
    if ($LASTEXITCODE -ne 0) { throw "Nu pot citi status in $path" }
    $dirtyCount = @($dirtyLines).Count

    $upstream = (& git -C $path rev-parse --abbrev-ref --symbolic-full-name "@{u}" 2>$null)
    $ahead = 0
    $behind = 0

    if ($LASTEXITCODE -eq 0 -and -not [string]::IsNullOrWhiteSpace(($upstream -join ""))) {
        $counts = (& git -C $path rev-list --left-right --count "$upstream...HEAD").Trim().Split(" ")
        if ($LASTEXITCODE -eq 0 -and $counts.Count -eq 2) {
            $behind = [int]$counts[0]
            $ahead = [int]$counts[1]
        }
    }

    $lastCommit = (& git -C $path log -1 --pretty=format:"%h %s").Trim()
    if ($LASTEXITCODE -ne 0) { $lastCommit = "-" }

    $rows += [PSCustomObject]@{
        agent      = $entry.agent
        branch     = $branch
        dirty      = $dirtyCount
        ahead      = $ahead
        behind     = $behind
        lastCommit = $lastCommit
    }
}

Write-Host ""
Write-Host "Status worktree-uri:"
$rows | Sort-Object agent | Format-Table -AutoSize

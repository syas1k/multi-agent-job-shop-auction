param(
    [string]$OutputDirectory = "data/benchmarks",
    [string]$Revision = ""
)
$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"
if (-not $Revision) {
    $Revision = (Invoke-RestMethod -Uri "https://api.github.com/repos/tamy0612/JSPLIB/commits/master" -TimeoutSec 30).sha
}
if ($Revision -notmatch '^[0-9a-f]{40}$') { throw "Expected a commit SHA" }
$base = "https://raw.githubusercontent.com/tamy0612/JSPLIB/$Revision/"
$metadata = Invoke-RestMethod -Uri ($base + "instances.json") -TimeoutSec 30
New-Item -ItemType Directory -Force $OutputDirectory | Out-Null
$records = @()
foreach ($name in @("ft06", "ft10", "ft20", "la01", "la16", "la31", "abz5", "ta51", "ta71")) {
    $item = $metadata | Where-Object { $_.name -eq $name }
    if (-not $item) { throw "Unknown instance $name" }
    $path = Join-Path $OutputDirectory "$name.txt"
    $url = $base + $item.path
    for ($attempt = 1; $attempt -le 3; $attempt++) {
        try {
            Invoke-WebRequest -Uri $url -OutFile $path -TimeoutSec 45
            break
        } catch {
            if ($attempt -eq 3) { throw }
            Write-Output "Retry $attempt for $name"
        }
    }
    $record = @{
        name=$name; jobs=$item.jobs; machines=$item.machines
        optimum=$item.optimum; file="$name.txt"; url=$url
        sha256=(Get-FileHash -LiteralPath $path -Algorithm SHA256).Hash.ToLower()
    }
    if ($item.bounds) { $record.bounds=$item.bounds }
    $records += $record
    Write-Output "$name : $($item.jobs)x$($item.machines)"
}
$manifest = @{repository="https://github.com/tamy0612/JSPLIB"; revision=$Revision; instances=$records}
$json = $manifest | ConvertTo-Json -Depth 8
[System.IO.File]::WriteAllText((Join-Path (Resolve-Path $OutputDirectory) "manifest.json"), $json, [System.Text.UTF8Encoding]::new($false))

[CmdletBinding()]
param(
    [string]$SshAlias = "school-gpu",
    [int]$LocalPort = 18001,
    [int]$RemotePort = 8001,
    [int]$NodeWaitSeconds = 1800,
    [int]$HealthWaitSeconds = 900,
    [string]$StateDirectory = ".runtime/real-llama-stack"
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$repositoryRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$statePath = Join-Path $repositoryRoot $StateDirectory
$nodePath = Join-Path $statePath "active_gpu_node.txt"
New-Item -ItemType Directory -Force -Path $statePath | Out-Null

$nodeDeadline = (Get-Date).AddSeconds($NodeWaitSeconds)
Write-Host "Waiting for the allocated compute-node marker..." -ForegroundColor Cyan
while (-not (Test-Path -LiteralPath $nodePath)) {
    if ((Get-Date) -ge $nodeDeadline) {
        throw "[TUNNEL FAILED] Timed out waiting for $nodePath."
    }
    Start-Sleep -Seconds 3
}

$computeNode = (Get-Content -LiteralPath $nodePath -Raw).Trim()
if ($computeNode -notmatch "^[A-Za-z0-9._-]+$") {
    throw "[TUNNEL FAILED] Invalid compute-node marker."
}

$existingListener = Get-NetTCPConnection -State Listen -LocalPort $LocalPort -ErrorAction SilentlyContinue
if ($existingListener) {
    throw "[TUNNEL FAILED] Local port $LocalPort is already in use."
}

$forward = "${LocalPort}:${computeNode}:${RemotePort}"
Write-Host "Opening 127.0.0.1:$LocalPort -> ${computeNode}:$RemotePort through $SshAlias" -ForegroundColor Cyan
$sshProcess = Start-Process -FilePath "ssh.exe" -NoNewWindow -PassThru -ArgumentList @(
    "-N", "-o", "BatchMode=yes", "-o", "ExitOnForwardFailure=yes", "-o", "ServerAliveInterval=30",
    "-o", "ServerAliveCountMax=3", "-L", $forward, $SshAlias
)

$healthUri = "http://127.0.0.1:$LocalPort/v1/models"
$healthDeadline = (Get-Date).AddSeconds($HealthWaitSeconds)
while ((Get-Date) -lt $healthDeadline) {
    if ($sshProcess.HasExited) {
        throw "[TUNNEL FAILED] ssh tunnel exited with code $($sshProcess.ExitCode)."
    }
    try {
        $response = Invoke-WebRequest -UseBasicParsing -Uri $healthUri -TimeoutSec 5
        if ($response.StatusCode -eq 200) {
            Write-Host "Tunnel and vLLM health are ready: $healthUri" -ForegroundColor Green
            Wait-Process -Id $sshProcess.Id
            exit $sshProcess.ExitCode
        }
    } catch {
        Start-Sleep -Seconds 5
    }
}

if (-not $sshProcess.HasExited) {
    Stop-Process -Id $sshProcess.Id
}
throw "[TUNNEL FAILED] TCP tunnel opened but vLLM health did not become ready within $HealthWaitSeconds seconds."

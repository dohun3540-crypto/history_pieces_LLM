[CmdletBinding()]
param(
    [string]$SshAlias = "school-gpu",
    [int]$LocalLlmPort = 18001,
    [int]$AppPort = 8000,
    [int]$NodeWaitSeconds = 1800,
    [int]$LlmWaitSeconds = 1200,
    [int]$AppWaitSeconds = 180
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$repositoryRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$statePath = Join-Path $repositoryRoot ".runtime\real-llama-stack"
$nodePath = Join-Path $statePath "active_gpu_node.txt"
$slurmJobPath = Join-Path $statePath "active_slurm_job.txt"
$pythonPath = Join-Path $repositoryRoot ".venv\Scripts\python.exe"
$roleProcesses = [System.Collections.Generic.List[System.Diagnostics.Process]]::new()

if (-not ("HistoryLauncherCancellation" -as [type])) {
    Add-Type -TypeDefinition @"
using System;

public static class HistoryLauncherCancellation
{
    private static ConsoleCancelEventHandler handler;
    private static volatile bool requested;

    public static bool IsRequested { get { return requested; } }

    public static void Register()
    {
        requested = false;
        handler = delegate(object sender, ConsoleCancelEventArgs args) {
            args.Cancel = true;
            requested = true;
        };
        Console.CancelKeyPress += handler;
    }

    public static void Unregister()
    {
        if (handler != null) {
            Console.CancelKeyPress -= handler;
            handler = null;
        }
    }
}
"@
}

function Assert-NotCancelled {
    if ([HistoryLauncherCancellation]::IsRequested) {
        throw [System.OperationCanceledException]::new("Launcher cancellation requested.")
    }
}

function Assert-RoleRunning {
    param([System.Diagnostics.Process]$Process, [string]$RoleName)

    if ($Process.HasExited) {
        $stdoutPath = Join-Path $statePath "$RoleName.stdout.log"
        $stderrPath = Join-Path $statePath "$RoleName.stderr.log"
        $details = [System.Collections.Generic.List[string]]::new()
        if (Test-Path -LiteralPath $stdoutPath) {
            $details.AddRange([string[]]@(Get-Content -LiteralPath $stdoutPath -Tail 30))
        }
        if (Test-Path -LiteralPath $stderrPath) {
            $details.AddRange([string[]]@(Get-Content -LiteralPath $stderrPath -Tail 30))
        }
        throw "[$RoleName FAILED] Background process exited with code $($Process.ExitCode).`n$($details -join [Environment]::NewLine)"
    }
}

function Start-RoleProcess {
    param([string]$RoleName, [string]$ScriptName, [string[]]$RoleArguments)

    $scriptPath = Join-Path $PSScriptRoot $ScriptName
    $stdoutPath = Join-Path $statePath "$RoleName.stdout.log"
    $stderrPath = Join-Path $statePath "$RoleName.stderr.log"
    $arguments = @("-NoProfile", "-ExecutionPolicy", "Bypass", "-File", $scriptPath) + $RoleArguments
    $process = Start-Process -FilePath "powershell.exe" -WorkingDirectory $repositoryRoot `
        -NoNewWindow -PassThru -ArgumentList $arguments `
        -RedirectStandardOutput $stdoutPath -RedirectStandardError $stderrPath
    $roleProcesses.Add($process)
    Write-Host "Started $RoleName in the background (PID $($process.Id))." -ForegroundColor Cyan
    return $process
}

function Stop-ProcessTree {
    param([int]$ProcessId)

    $children = @(Get-CimInstance Win32_Process -Filter "ParentProcessId = $ProcessId" -ErrorAction SilentlyContinue)
    foreach ($child in $children) {
        Stop-ProcessTree -ProcessId ([int]$child.ProcessId)
    }
    Stop-Process -Id $ProcessId -Force -ErrorAction SilentlyContinue
}

New-Item -ItemType Directory -Force -Path $statePath | Out-Null
foreach ($runtimeFile in @(
    "active_gpu_node.txt", "active_slurm_job.txt", "remote-vllm.log",
    "remote.stdout.log", "remote.stderr.log",
    "tunnel.stdout.log", "tunnel.stderr.log",
    "app.stdout.log", "app.stderr.log"
)) {
    Remove-Item -LiteralPath (Join-Path $statePath $runtimeFile) -Force -ErrorAction SilentlyContinue
}

if (-not (Get-Command ssh.exe -ErrorAction SilentlyContinue)) {
    throw "OpenSSH client was not found."
}
if (-not (Test-Path -LiteralPath $pythonPath -PathType Leaf)) {
    throw "Project Python was not found: $pythonPath"
}
if (Get-NetTCPConnection -State Listen -LocalPort $LocalLlmPort -ErrorAction SilentlyContinue) {
    throw "Local LLM port $LocalLlmPort is already in use."
}
if (Get-NetTCPConnection -State Listen -LocalPort $AppPort -ErrorAction SilentlyContinue) {
    throw "Local app port $AppPort is already in use."
}

Write-Host "Checking SSH key authentication for $SshAlias ..." -ForegroundColor Cyan
$sshCheck = & ssh.exe -o BatchMode=yes -o ConnectTimeout=10 $SshAlias "echo SSH_KEY_OK" 2>&1
if ($LASTEXITCODE -ne 0 -or ($sshCheck -join "`n") -notmatch "SSH_KEY_OK") {
    throw "[SSH KEY AUTH FAILED] $($sshCheck -join [Environment]::NewLine)"
}
Write-Host "SSH key authentication succeeded." -ForegroundColor Green

[HistoryLauncherCancellation]::Register()
try {
    $remoteProcess = Start-RoleProcess -RoleName "remote" -ScriptName "start_remote_vllm.ps1" `
        -RoleArguments @("-SshAlias", $SshAlias)

    Write-Host "Waiting for Slurm to report the allocated compute node ..." -ForegroundColor Cyan
    $nodeDeadline = (Get-Date).AddSeconds($NodeWaitSeconds)
    while (-not (Test-Path -LiteralPath $nodePath)) {
        Assert-NotCancelled
        Assert-RoleRunning -Process $remoteProcess -RoleName "remote"
        if ((Get-Date) -ge $nodeDeadline) {
            throw "[SLURM FAILED] Timed out waiting for compute-node discovery."
        }
        Start-Sleep -Seconds 3
    }
    $computeNode = (Get-Content -LiteralPath $nodePath -Raw).Trim()
    if ($computeNode -notmatch "^[A-Za-z0-9._-]+$") {
        throw "[SLURM FAILED] Invalid compute-node marker: $computeNode"
    }
    Write-Host "Slurm allocated compute node: $computeNode" -ForegroundColor Green

    $tunnelProcess = Start-RoleProcess -RoleName "tunnel" -ScriptName "start_vllm_tunnel.ps1" `
        -RoleArguments @("-SshAlias", $SshAlias, "-LocalPort", [string]$LocalLlmPort)

    $modelsUri = "http://127.0.0.1:$LocalLlmPort/v1/models"
    Write-Host "Waiting for real vLLM model discovery through the SSH tunnel ..." -ForegroundColor Cyan
    $llmDeadline = (Get-Date).AddSeconds($LlmWaitSeconds)
    $servedModel = ""
    while ((Get-Date) -lt $llmDeadline) {
        Assert-NotCancelled
        Assert-RoleRunning -Process $remoteProcess -RoleName "remote"
        Assert-RoleRunning -Process $tunnelProcess -RoleName "tunnel"
        try {
            $models = Invoke-RestMethod -Method Get -Uri $modelsUri -TimeoutSec 5
            $servedModel = [string]$models.data[0].id
            if (-not [string]::IsNullOrWhiteSpace($servedModel)) { break }
        } catch {
            Start-Sleep -Seconds 5
        }
    }
    if ([string]::IsNullOrWhiteSpace($servedModel)) {
        throw "[VLLM FAILED] /v1/models did not become ready within $LlmWaitSeconds seconds."
    }
    Write-Host "vLLM is ready with real model: $servedModel" -ForegroundColor Green

    $appProcess = Start-RoleProcess -RoleName "app" -ScriptName "start_history_app.ps1" `
        -RoleArguments @("-LlmBaseUrl", "http://127.0.0.1:$LocalLlmPort", "-AppPort", [string]$AppPort)

    $readyUri = "http://127.0.0.1:$AppPort/ready"
    Write-Host "Waiting for generation probe and History Chatbot readiness ..." -ForegroundColor Cyan
    $appDeadline = (Get-Date).AddSeconds($AppWaitSeconds)
    $ready = $null
    while ((Get-Date) -lt $appDeadline) {
        Assert-NotCancelled
        Assert-RoleRunning -Process $remoteProcess -RoleName "remote"
        Assert-RoleRunning -Process $tunnelProcess -RoleName "tunnel"
        Assert-RoleRunning -Process $appProcess -RoleName "app"
        try {
            $ready = Invoke-RestMethod -Method Get -Uri $readyUri -TimeoutSec 5
            if ($ready.ready -and $ready.retriever -and $ready.llm) { break }
        } catch {
            Start-Sleep -Seconds 3
        }
    }
    if (-not $ready -or -not $ready.ready -or -not $ready.retriever -or -not $ready.llm) {
        throw "[PROJECT START FAILED] /ready did not report retriever=true and llm=true."
    }

    $browserUri = "http://127.0.0.1:$AppPort/"
    Write-Host "History Chatbot is ready: $browserUri" -ForegroundColor Green
    Write-Host "Logs: $statePath" -ForegroundColor DarkGray
    Write-Host "Keep this supervisor open; press Ctrl+C to stop this stack." -ForegroundColor Yellow
    Start-Process $browserUri | Out-Null

    while ($true) {
        Assert-NotCancelled
        Assert-RoleRunning -Process $remoteProcess -RoleName "remote"
        Assert-RoleRunning -Process $tunnelProcess -RoleName "tunnel"
        Assert-RoleRunning -Process $appProcess -RoleName "app"
        Start-Sleep -Seconds 3
    }
} catch [System.OperationCanceledException] {
    Write-Host "Ctrl+C received; shutting down the launcher-owned stack." -ForegroundColor Yellow
} finally {
    if ($roleProcesses.Count -gt 0) {
        Write-Host "Stopping launcher-owned local processes and the remote Slurm step ..." -ForegroundColor Yellow
    }
    foreach ($process in (@($roleProcesses) | Sort-Object Id -Descending)) {
        if (-not $process.HasExited) {
            Stop-ProcessTree -ProcessId $process.Id
        }
    }
    if (Test-Path -LiteralPath $slurmJobPath -PathType Leaf) {
        $slurmJobId = (Get-Content -LiteralPath $slurmJobPath -Raw).Trim()
        if ($slurmJobId -match "^[0-9]+$") {
            Write-Host "Cancelling launcher-owned Slurm job $slurmJobId ..." -ForegroundColor Yellow
            $cancelInfo = [System.Diagnostics.ProcessStartInfo]::new()
            $cancelInfo.FileName = (Get-Command ssh.exe -ErrorAction Stop).Source
            $cancelInfo.Arguments = "-o BatchMode=yes $SshAlias `"scancel $slurmJobId`""
            $cancelInfo.UseShellExecute = $false
            $cancelInfo.CreateNoWindow = $true
            $cancelProcess = [System.Diagnostics.Process]::new()
            $cancelProcess.StartInfo = $cancelInfo
            if ($cancelProcess.Start()) {
                $cancelProcess.WaitForExit()
                if ($cancelProcess.ExitCode -ne 0) {
                    Write-Warning "Could not cancel launcher-owned Slurm job $slurmJobId (SSH exit $($cancelProcess.ExitCode))."
                }
            }
        }
    }
    [HistoryLauncherCancellation]::Unregister()
}

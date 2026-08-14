[CmdletBinding()]
param(
    [string]$SshAlias = "school-gpu",
    [string]$Partition = "p02",
    [string]$GpuResource = "gpu:1",
    [string]$AllocationTime = "04:00:00",
    [string]$VllmExecutable = "/abr/jn_hack01/vllm-cu121-clean/bin/vllm",
    [string]$ModelPath = "/abr/jn_hack01/.cache/huggingface/hub/models--beomi--Llama-3-Open-Ko-8B-Instruct-preview/snapshots/d8c93440c5c0426f0127e2baf822ce5b60fa3a73",
    [string]$ServedModelName = "beomi/Llama-3-Open-Ko-8B-Instruct-preview",
    [string]$StateDirectory = ".runtime/real-llama-stack"
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$repositoryRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$statePath = Join-Path $repositoryRoot $StateDirectory
$nodePath = Join-Path $statePath "active_gpu_node.txt"
$slurmJobPath = Join-Path $statePath "active_slurm_job.txt"
$remoteLogPath = Join-Path $statePath "remote-vllm.log"
$sshStdoutPath = Join-Path $statePath "ssh.stdout.log"
$sshStderrPath = Join-Path $statePath "ssh.stderr.log"
$sshWrapperPath = Join-Path $statePath "run-ssh.cmd"
New-Item -ItemType Directory -Force -Path $statePath | Out-Null
Remove-Item -LiteralPath $nodePath, $slurmJobPath, $remoteLogPath, $sshStdoutPath, $sshStderrPath, $sshWrapperPath `
    -Force -ErrorAction SilentlyContinue

if (-not (Get-Command ssh.exe -ErrorAction SilentlyContinue)) {
    throw "[REMOTE GPU CONNECTION BLOCKED] ssh executable was not found."
}
foreach ($value in @($SshAlias, $Partition, $GpuResource, $AllocationTime, $VllmExecutable, $ModelPath, $ServedModelName)) {
    if ($value -notmatch "^[A-Za-z0-9_./:-]+$") {
        throw "Remote launch parameter contains unsupported shell characters: $value"
    }
}

$innerScript = @'
set -euo pipefail

echo "HISTORY_GPU_NODE=$(hostname)"
echo "HISTORY_SLURM_JOB_ID=${SLURM_JOB_ID:?}"
echo "HISTORY_GPU_ADDRESSES=$(hostname -I 2>/dev/null || true)"

if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "[VLLM START FAILED] nvidia-smi is unavailable on the allocated node." >&2
    exit 20
fi
nvidia-smi -L

VLLM_EXECUTABLE=__VLLM_EXECUTABLE__
MODEL_PATH=__MODEL_PATH__
SERVED_MODEL=__SERVED_MODEL__
if [ ! -d "$MODEL_PATH" ]; then
    echo "[VLLM ENVIRONMENT BLOCKED] Confirmed model checkpoint is absent: $MODEL_PATH" >&2
    exit 21
fi

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

if [ -x "$VLLM_EXECUTABLE" ]; then
    echo "HISTORY_VLLM_EXECUTABLE=$VLLM_EXECUTABLE"
    "$VLLM_EXECUTABLE" --version 2>/dev/null || true
    exec "$VLLM_EXECUTABLE" serve "$MODEL_PATH" \
        --served-model-name "$SERVED_MODEL" \
        --host 0.0.0.0 \
        --port 8001 \
        --dtype bfloat16 \
        --max-model-len 8192 \
        --gpu-memory-utilization 0.85
fi

echo "[VLLM ENVIRONMENT BLOCKED] Required vLLM executable is absent or not executable: $VLLM_EXECUTABLE" >&2
exit 22
'@

$innerScript = $innerScript.Replace("__VLLM_EXECUTABLE__", "'$VllmExecutable'")
$innerScript = $innerScript.Replace("__MODEL_PATH__", "'$ModelPath'")
$innerScript = $innerScript.Replace("__SERVED_MODEL__", "'$ServedModelName'")
$innerBase64 = [Convert]::ToBase64String([System.Text.Encoding]::UTF8.GetBytes($innerScript))
$remoteCommand = "command -v srun >/dev/null 2>&1 || { echo '[REMOTE GPU CONNECTION BLOCKED] srun is unavailable.' >&2; exit 10; }; " +
    "srun --partition=$Partition --gres=$GpuResource --nodes=1 --ntasks=1 --time=$AllocationTime " +
    "--unbuffered --kill-on-bad-exit " +
    "bash -lc 'echo $innerBase64 | base64 -d | bash'"

Write-Host "Connecting with SSH key authentication and requesting a non-interactive Slurm GPU job ..." -ForegroundColor Cyan

function Publish-SshOutput {
    param(
        [string]$Path,
        [ref]$Cursor,
        [string]$StreamName
    )

    if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) {
        return
    }
    $lines = @(Get-Content -LiteralPath $Path)
    for ($index = [int]$Cursor.Value; $index -lt $lines.Count; $index++) {
        $line = [string]$lines[$index]
        $logLine = if ($StreamName -eq "stderr") { "[stderr] $line" } else { $line }
        Add-Content -LiteralPath $remoteLogPath -Value $logLine
        Write-Output $logLine

        $plain = $line -replace "`e\[[0-9;?]*[ -/]*[@-~]", ""
        if ($plain -match "HISTORY_GPU_NODE=([A-Za-z0-9._-]+)") {
            Set-Content -LiteralPath $nodePath -Value $Matches[1] -Encoding ascii
            Write-Output "Recorded allocated compute node: $($Matches[1])"
        }
        if ($plain -match "HISTORY_SLURM_JOB_ID=([0-9]+)") {
            Set-Content -LiteralPath $slurmJobPath -Value $Matches[1] -Encoding ascii
            Write-Output "Recorded launcher-owned Slurm job: $($Matches[1])"
        }
    }
    $Cursor.Value = $lines.Count
}

$sshPath = (Get-Command ssh.exe -ErrorAction Stop).Source
if ($remoteCommand.Contains('"')) {
    throw "Generated remote command contains an unsupported double quote."
}
$sshInvocation = "`"$sshPath`" -o BatchMode=yes -o ServerAliveInterval=30 " +
    "-o ServerAliveCountMax=3 $SshAlias `"$remoteCommand`" " +
    "1>`"$sshStdoutPath`" 2>`"$sshStderrPath`""
$sshWrapper = @("@echo off", $sshInvocation, "exit /b %ERRORLEVEL%")
Set-Content -LiteralPath $sshWrapperPath -Value $sshWrapper -Encoding ascii

$stdoutCursor = 0
$stderrCursor = 0
$sshProcess = $null
$sshExitCode = $null

try {
    $startInfo = [System.Diagnostics.ProcessStartInfo]::new()
    $startInfo.FileName = $env:ComSpec
    $startInfo.Arguments = "/d /s /c `"`"$sshWrapperPath`"`""
    $startInfo.UseShellExecute = $false
    $startInfo.CreateNoWindow = $true
    $sshProcess = [System.Diagnostics.Process]::new()
    $sshProcess.StartInfo = $startInfo
    if (-not $sshProcess.Start()) {
        throw "[VLLM START FAILED] Could not start the SSH process wrapper."
    }

    while (-not $sshProcess.HasExited) {
        Publish-SshOutput -Path $sshStdoutPath -Cursor ([ref]$stdoutCursor) -StreamName "stdout"
        Publish-SshOutput -Path $sshStderrPath -Cursor ([ref]$stderrCursor) -StreamName "stderr"
        Start-Sleep -Milliseconds 500
        $sshProcess.Refresh()
    }
    $sshProcess.WaitForExit()
    Publish-SshOutput -Path $sshStdoutPath -Cursor ([ref]$stdoutCursor) -StreamName "stdout"
    Publish-SshOutput -Path $sshStderrPath -Cursor ([ref]$stderrCursor) -StreamName "stderr"
    $sshExitCode = $sshProcess.ExitCode
} catch {
    if ($null -ne $sshProcess -and -not $sshProcess.HasExited) {
        Stop-Process -Id $sshProcess.Id -Force -ErrorAction SilentlyContinue
    }
    throw
} finally {
    if ($sshExitCode -eq 0) {
        Remove-Item -LiteralPath $sshWrapperPath -Force -ErrorAction SilentlyContinue
    }
}

if ($sshExitCode -ne 0) {
    throw "[VLLM START FAILED] Remote SSH/Slurm/vLLM process exited with code $sshExitCode."
}

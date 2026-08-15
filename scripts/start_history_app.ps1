[CmdletBinding()]
param(
    [string]$LlmBaseUrl = "http://127.0.0.1:18001",
    [int]$AppPort = 8000,
    [int]$LlmWaitSeconds = 1200,
    [int]$AppWaitSeconds = 180
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$repositoryRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$pythonPath = Join-Path $repositoryRoot ".venv\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $pythonPath -PathType Leaf)) {
    throw "[PROJECT START FAILED] Project Python was not found: $pythonPath"
}

$headers = @{}
if (-not [string]::IsNullOrWhiteSpace([string]$env:HISTORY_LLM_API_KEY)) {
    $headers.Authorization = "Bearer $env:HISTORY_LLM_API_KEY"
}

$baseUrl = $LlmBaseUrl.TrimEnd("/")
$modelsUri = "$baseUrl/v1/models"
$deadline = (Get-Date).AddSeconds($LlmWaitSeconds)
$servedModel = ""
Write-Host "Waiting for real vLLM model discovery at $modelsUri ..." -ForegroundColor Cyan
while ((Get-Date) -lt $deadline) {
    try {
        $models = Invoke-RestMethod -Method Get -Uri $modelsUri -Headers $headers -TimeoutSec 10
        $servedModel = [string]$models.data[0].id
        if (-not [string]::IsNullOrWhiteSpace($servedModel)) { break }
    } catch {
        Start-Sleep -Seconds 5
    }
}
if ([string]::IsNullOrWhiteSpace($servedModel)) {
    throw "[PROJECT START FAILED] vLLM model discovery timed out."
}

# Windows PowerShell 5.1 can decode a BOM-less UTF-8 .ps1 file as ANSI.  Keep
# the probe payload ASCII-safe while still sending the intended UTF-8 text.
$readinessProbeMessage = [System.Text.Encoding]::UTF8.GetString(
    [System.Convert]::FromBase64String(
        "7Jew6rKwIO2ZleyduOydtOudvOqzoCDsp6fqsowg64u17ZW0IOyjvOyEuOyalC4="
    )
)
$probeBody = @{
    model = $servedModel
    messages = @(@{ role = "user"; content = $readinessProbeMessage })
    temperature = 0.2
    top_p = 0.9
    max_tokens = 32
    stream = $false
} | ConvertTo-Json -Depth 5 -Compress
$probe = Invoke-RestMethod -Method Post -Uri "$baseUrl/v1/chat/completions" `
    -Headers $headers -ContentType "application/json; charset=utf-8" `
    -Body ([System.Text.Encoding]::UTF8.GetBytes($probeBody)) -TimeoutSec 60
$completion = [string]$probe.choices[0].message.content
if ([string]::IsNullOrWhiteSpace($completion)) {
    throw "[PROJECT START FAILED] Actual vLLM completion was empty."
}
Write-Host "Actual Llama completion succeeded with model: $servedModel" -ForegroundColor Green

$env:PYTHONPATH = "src"
$env:HISTORY_LLM_BACKEND = "openai_compatible"
$env:HISTORY_LLM_BASE_URL = $baseUrl
$env:HISTORY_LLM_MODEL_ID = $servedModel
$env:HISTORY_LLM_ALLOWED_HOSTS = "127.0.0.1"
$env:LLM_READINESS_PROBE = "true"
$env:LLM_MAX_NEW_TOKENS = "256"
$env:APP_MODE = "hackathon"

$existingListener = Get-NetTCPConnection -State Listen -LocalPort $AppPort -ErrorAction SilentlyContinue
if ($existingListener) {
    throw "[PROJECT START FAILED] Local port $AppPort is already in use."
}

$arguments = @(
    "-m", "uvicorn", "history_chatbot.chat.api:create_app", "--factory",
    "--host", "127.0.0.1", "--port", [string]$AppPort
)
$appProcess = Start-Process -FilePath $pythonPath -WorkingDirectory $repositoryRoot `
    -NoNewWindow -PassThru -ArgumentList $arguments
$readyUri = "http://127.0.0.1:$AppPort/ready"
$readyDeadline = (Get-Date).AddSeconds($AppWaitSeconds)
while ((Get-Date) -lt $readyDeadline) {
    if ($appProcess.HasExited) {
        throw "[PROJECT START FAILED] FastAPI exited with code $($appProcess.ExitCode)."
    }
    try {
        $ready = Invoke-RestMethod -Method Get -Uri $readyUri -TimeoutSec 5
        if ($ready.ready -and $ready.retriever -and $ready.llm) {
            Write-Host "Project is ready with real LLM: $readyUri" -ForegroundColor Green
            Write-Host "Browser UI: http://127.0.0.1:$AppPort/" -ForegroundColor Green
            Wait-Process -Id $appProcess.Id
            exit $appProcess.ExitCode
        }
    } catch {
        Start-Sleep -Seconds 3
    }
}

if (-not $appProcess.HasExited) {
    Stop-Process -Id $appProcess.Id
}
throw "[PROJECT START FAILED] FastAPI did not report retriever=true and llm=true within $AppWaitSeconds seconds."

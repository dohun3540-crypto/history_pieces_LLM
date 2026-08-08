[CmdletBinding()]
param(
    [string]$AppBaseUrl = "http://127.0.0.1:8000",
    [string]$EvidenceQuestion = "목포 개항 이후 도시에는 어떤 변화가 있었나요?",
    [string[]]$FallbackProbeQuestions = @(
        "양자컴퓨터의 큐비트 오류 정정 방법을 설명해 주세요."
    )
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
$script:Failures = 0

function Write-Pass([string]$Message) {
    Write-Host "PASS: $Message" -ForegroundColor Green
}

function Write-Fail([string]$Message) {
    $script:Failures += 1
    Write-Host "FAIL: $Message" -ForegroundColor Red
}

function Write-Skip([string]$Message) {
    Write-Host "SKIP: $Message" -ForegroundColor Yellow
}

function ConvertTo-Utf8JsonBytes([object]$Value) {
    $json = $Value | ConvertTo-Json -Depth 8 -Compress
    return ,([System.Text.Encoding]::UTF8.GetBytes([string]$json))
}

function Invoke-JsonPost([string]$Uri, [object]$Value, [hashtable]$Headers) {
    [byte[]]$body = ConvertTo-Utf8JsonBytes $Value
    return Invoke-RestMethod `
        -Method Post `
        -Uri $Uri `
        -Headers $Headers `
        -ContentType "application/json; charset=utf-8" `
        -Body $body
}

$llmBaseUrl = [string]$env:HISTORY_LLM_BASE_URL
$modelId = [string]$env:HISTORY_LLM_MODEL_ID
if ([string]::IsNullOrWhiteSpace($llmBaseUrl)) {
    Write-Fail "HISTORY_LLM_BASE_URL is not set"
}
if ([string]::IsNullOrWhiteSpace($modelId)) {
    Write-Fail "HISTORY_LLM_MODEL_ID is not set"
}
if ($script:Failures -gt 0) {
    exit 1
}

$llmBaseUrl = $llmBaseUrl.TrimEnd("/")
$AppBaseUrl = $AppBaseUrl.TrimEnd("/")
$llmHeaders = @{}
if (-not [string]::IsNullOrWhiteSpace([string]$env:HISTORY_LLM_API_KEY)) {
    $llmHeaders.Authorization = "Bearer $env:HISTORY_LLM_API_KEY"
}

$remoteHealthReady = $false
$remoteModelReady = $false
$remoteChatReady = $false

try {
    $health = Invoke-WebRequest `
        -UseBasicParsing `
        -Method Get `
        -Uri "$llmBaseUrl/health" `
        -Headers $llmHeaders
    if ($health.StatusCode -ne 200) {
        throw "HTTP $($health.StatusCode)"
    }
    $remoteHealthReady = $true
    Write-Pass "remote GET /health"
} catch {
    Write-Fail "remote GET /health - $($_.Exception.Message)"
}

if ($remoteHealthReady) {
    try {
        $models = Invoke-RestMethod `
            -Method Get `
            -Uri "$llmBaseUrl/v1/models" `
            -Headers $llmHeaders
        $servedIds = @($models.data | ForEach-Object { [string]$_.id })
        if ($servedIds -notcontains $modelId) {
            throw "configured model is not served"
        }
        $remoteModelReady = $true
        Write-Pass "remote GET /v1/models contains configured model"
    } catch {
        Write-Fail "remote GET /v1/models - $($_.Exception.Message)"
    }
} else {
    Write-Skip "remote /v1/models because /health failed"
}

if ($remoteModelReady) {
    try {
        $direct = Invoke-JsonPost `
            -Uri "$llmBaseUrl/v1/chat/completions" `
            -Headers $llmHeaders `
            -Value @{
                model = $modelId
                messages = @(
                    @{ role = "system"; content = "한국어로 한 문장만 답하세요." }
                    @{ role = "user"; content = "연결 확인이라고 답하세요." }
                )
                max_tokens = 32
                temperature = 0.0
                stream = $false
            }
        $directContent = [string]$direct.choices[0].message.content
        if ([string]::IsNullOrWhiteSpace($directContent)) {
            throw "empty completion"
        }
        $remoteChatReady = $true
        Write-Pass "remote POST /v1/chat/completions"
    } catch {
        Write-Fail "remote POST /v1/chat/completions - $($_.Exception.Message)"
    }
} else {
    Write-Skip "remote chat completion because configured model is not ready"
}

$localReady = $false
try {
    $ready = Invoke-RestMethod -Method Get -Uri "$AppBaseUrl/ready"
    if (-not $ready.ready -or -not $ready.retriever -or -not $ready.llm) {
        throw ($ready | ConvertTo-Json -Compress)
    }
    $localReady = $true
    Write-Pass "local GET /ready"
} catch {
    Write-Fail "local GET /ready - $($_.Exception.Message)"
}

try {
    $readyAlias = Invoke-RestMethod -Method Get -Uri "$AppBaseUrl/api/v1/ready"
    if (-not $readyAlias.ready -or -not $readyAlias.retriever -or -not $readyAlias.llm) {
        throw ($readyAlias | ConvertTo-Json -Compress)
    }
    Write-Pass "local GET /api/v1/ready"
} catch {
    Write-Fail "local GET /api/v1/ready - $($_.Exception.Message)"
}

$evidenceSearchReady = $false
try {
    $search = Invoke-JsonPost `
        -Uri "$AppBaseUrl/api/v1/search" `
        -Headers @{} `
        -Value @{ query = $EvidenceQuestion; top_k = 3 }
    if (@($search.results).Count -eq 0) {
        throw "no results for evidence question"
    }
    $evidenceSearchReady = $true
    Write-Pass "local POST /api/v1/search returned evidence"
} catch {
    Write-Fail "local POST /api/v1/search - $($_.Exception.Message)"
}

if ($localReady -and $remoteChatReady -and $evidenceSearchReady) {
    try {
        $answer = Invoke-JsonPost `
            -Uri "$AppBaseUrl/api/v1/chat" `
            -Headers @{} `
            -Value @{ message = $EvidenceQuestion }
        $answerFields = @($answer.PSObject.Properties.Name)
        if ($answerFields.Count -ne 1 -or $answerFields[0] -ne "answer") {
            throw "unexpected response fields: $($answerFields -join ',')"
        }
        if ([string]::IsNullOrWhiteSpace([string]$answer.answer)) {
            throw "empty answer"
        }
        Write-Pass "local POST /api/v1/chat returned answer-only response"
        Write-Host "PASS: verify one new POST /v1/chat/completions in the vLLM access log"
    } catch {
        Write-Fail "local evidence POST /api/v1/chat - $($_.Exception.Message)"
    }
} else {
    Write-Skip "local evidence chat because a prerequisite failed"
}

$fallbackQuestion = $null
foreach ($candidate in $FallbackProbeQuestions) {
    if ([string]::IsNullOrWhiteSpace($candidate)) {
        continue
    }
    try {
        $candidateSearch = Invoke-JsonPost `
            -Uri "$AppBaseUrl/api/v1/search" `
            -Headers @{} `
            -Value @{ query = $candidate; top_k = 3 }
        if (@($candidateSearch.results).Count -eq 0) {
            $fallbackQuestion = $candidate
            break
        }
    } catch {
        Write-Fail "fallback probe search - $($_.Exception.Message)"
        break
    }
}

if ([string]::IsNullOrWhiteSpace([string]$fallbackQuestion)) {
    Write-Skip "no zero-result fallback probe available for current index"
} elseif (-not $localReady) {
    Write-Skip "fallback chat because local API is not ready"
} else {
    try {
        $fallback = Invoke-JsonPost `
            -Uri "$AppBaseUrl/api/v1/chat" `
            -Headers @{} `
            -Value @{ message = $fallbackQuestion }
        if ([string]$fallback.answer -ne "제공된 역사 자료에서 충분한 근거를 찾지 못했습니다.") {
            throw "unexpected fallback answer"
        }
        Write-Pass "local fallback response"
        Write-Host "PASS: verify no new POST /v1/chat/completions in the vLLM access log"
    } catch {
        Write-Fail "local fallback POST /api/v1/chat - $($_.Exception.Message)"
    }
}

if ($script:Failures -gt 0) {
    Write-Host "FAIL: smoke test completed with $script:Failures failure(s)" -ForegroundColor Red
    exit 1
}

Write-Pass "smoke test completed"
exit 0

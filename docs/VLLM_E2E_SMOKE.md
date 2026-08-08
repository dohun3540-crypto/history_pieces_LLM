# vLLM End-to-End Smoke Test

이 절차는 Windows의 로컬 RAG가 선택한 최소 context만 SSH tunnel을 통해 원격
vLLM에 전송하는지 확인한다. 실제 IP, SSH 사용자명, 비밀번호, served model name,
API token은 저장소에 기록하지 않는다. Windows에 vLLM이나 Llama 모델을 설치하지
않는다.

## 1. Slurm job과 SSH tunnel

전제 조건은 다음과 같다.

- GPU 노드에 active Slurm interactive job이 있다.
- active job 소유자에 대해 `pam_slurm_adopt`가 GPU 노드 SSH를 허용한다.
- vLLM은 GPU 노드의 `127.0.0.1:<REMOTE_PORT>`에 bind되어 있다.
- login node SSH port와 GPU node SSH port를 각각 확인했다.

이 구조에서는 Windows에서 login node를 ProxyJump로 거쳐 GPU node에 SSH로
접속하고, 최종 GPU node의 loopback으로 port forwarding하는 방식을 우선한다.
login node가 non-default SSH port를 사용하므로 `<LOGIN_PORT>`를 반드시 지정한다.
`-p <GPU_SSH_PORT>`는 최종 GPU node에 적용되는 port다.

```powershell
ssh -N `
  -J <SSH_USER>@<LOGIN_HOST>:<LOGIN_PORT> `
  -p <GPU_SSH_PORT> `
  -L 8001:127.0.0.1:<REMOTE_PORT> `
  <SSH_USER>@<GPU_NODE>
```

이 명령은 active Slurm job이 유지되고 GPU node SSH가 허용되는 동안에만 성공한다.
job이 종료되거나 `pam_slurm_adopt` 조건을 만족하지 않으면 새 SSH 연결 및 기존
tunnel이 종료될 수 있다.

SSH config를 사용하는 경우 port 역할을 명확히 분리한다. 다음 내용은 사용자의 로컬
`~/.ssh/config`에만 두며 저장소에 실제 값을 기록하지 않는다.

```sshconfig
Host <LOGIN_ALIAS>
    HostName <LOGIN_HOST>
    User <SSH_USER>
    Port <LOGIN_PORT>

Host <GPU_ALIAS>
    HostName <GPU_NODE>
    User <SSH_USER>
    Port <GPU_SSH_PORT>
    ProxyJump <LOGIN_ALIAS>
```

SSH config가 준비된 경우 tunnel 명령은 다음과 같다.

```powershell
ssh -N -L 8001:127.0.0.1:<REMOTE_PORT> <GPU_ALIAS>
```

다음 형태는 vLLM이 GPU node의 `127.0.0.1`에만 bind된 현재 구조에서 사용하면
안 된다.

```powershell
# 사용 금지: login node에서 GPU node의 loopback service에 도달하지 못한다.
ssh -N -p <LOGIN_PORT> `
  -L 8001:<GPU_NODE>:<REMOTE_PORT> `
  <SSH_USER>@<LOGIN_HOST>
```

위 명령의 forwarding 연결은 login node에서 `<GPU_NODE>:<REMOTE_PORT>`로
시작된다. 따라서 GPU node loopback에만 bind된 vLLM에는 연결되지 않는다. 이를
해결하려고 vLLM을 무조건 `0.0.0.0`에 공개하지 않는다. 우선 절차인 GPU node까지의
ProxyJump SSH tunnel을 사용한다.

tunnel을 연 뒤 다른 PowerShell에서 확인한다.

```powershell
Test-NetConnection 127.0.0.1 -Port 8001
```

## 2. 로컬 환경변수

저장소 루트 `C:\projects\history_pieces_LLM`에서 설정한다. `<served-model-name>`은
원격 vLLM의 `--served-model-name` 및 `/v1/models`의 `data[].id`와 정확히 같아야
한다.

```powershell
$env:PYTHONPATH = "src"
$env:APP_MODE = "hackathon"

$env:HISTORY_LLM_BACKEND = "openai_compatible"
$env:HISTORY_LLM_BASE_URL = "http://127.0.0.1:8001"
$env:HISTORY_LLM_MODEL_ID = "<served-model-name>"
$env:HISTORY_LLM_API_FORMAT = "openai"

$env:LLM_READINESS_PROBE = "true"
$env:LLM_TIMEOUT_SECONDS = "120"
$env:LLM_MAX_NEW_TOKENS = "256"
$env:LLM_REMOTE_HISTORY_ENABLED = "false"
$env:LLM_REMOTE_SANITIZE_ENABLED = "true"
```

vLLM을 token으로 보호한 경우 Windows PowerShell 5.1에서 process-scoped 값만 입력한다.

```powershell
$env:LLM_API_KEY_REQUIRED = "true"
$secureToken = Read-Host "Temporary vLLM API token" -AsSecureString
$tokenPointer = [Runtime.InteropServices.Marshal]::SecureStringToBSTR($secureToken)
try {
    $env:HISTORY_LLM_API_KEY = [Runtime.InteropServices.Marshal]::PtrToStringBSTR($tokenPointer)
} finally {
    [Runtime.InteropServices.Marshal]::ZeroFreeBSTR($tokenPointer)
    Remove-Variable secureToken, tokenPointer -ErrorAction SilentlyContinue
}
```

token을 사용하지 않는 서버에서는 값을 남기지 않는다.

```powershell
$env:LLM_API_KEY_REQUIRED = "false"
Remove-Item Env:HISTORY_LLM_API_KEY -ErrorAction SilentlyContinue
```

## 3. vLLM 직접 검사

```powershell
$llmHeaders = @{}
if ($env:HISTORY_LLM_API_KEY) {
    $llmHeaders.Authorization = "Bearer $env:HISTORY_LLM_API_KEY"
}

$health = Invoke-WebRequest `
    -Method Get `
    -Uri "$env:HISTORY_LLM_BASE_URL/health" `
    -Headers $llmHeaders `
    -UseBasicParsing
if ($health.StatusCode -ne 200) { throw "vLLM health failed" }

$models = Invoke-RestMethod `
    -Method Get `
    -Uri "$env:HISTORY_LLM_BASE_URL/v1/models" `
    -Headers $llmHeaders
$servedIds = @($models.data | ForEach-Object { $_.id })
if ($servedIds -notcontains $env:HISTORY_LLM_MODEL_ID) {
    throw "Configured model is not in /v1/models"
}
```

OpenAI-compatible chat completion을 직접 한 번 확인한다.

```powershell
$directBody = @{
    model = $env:HISTORY_LLM_MODEL_ID
    messages = @(
        @{ role = "system"; content = "한국어로 한 문장만 답하세요." }
        @{ role = "user"; content = "연결 확인이라고 답하세요." }
    )
    max_tokens = 32
    temperature = 0.0
    stream = $false
} | ConvertTo-Json -Depth 5

$direct = Invoke-RestMethod `
    -Method Post `
    -Uri "$env:HISTORY_LLM_BASE_URL/v1/chat/completions" `
    -Headers $llmHeaders `
    -ContentType "application/json; charset=utf-8" `
    -Body $directBody
if (-not $direct.choices[0].message.content) {
    throw "vLLM returned an empty chat completion"
}
$direct.choices[0].message.content
```

## 4. 로컬 FastAPI 실행과 readiness

환경변수를 설정한 같은 PowerShell에서 실행한다.

```powershell
python -m uvicorn history_chatbot.chat.api:create_app `
  --factory `
  --host 127.0.0.1 `
  --port 8000
```

다른 PowerShell에서 로컬 endpoint를 순서대로 확인한다.

```powershell
$appBaseUrl = "http://127.0.0.1:8000"

$ready = Invoke-RestMethod -Method Get -Uri "$appBaseUrl/ready"
if (-not $ready.retriever -or -not $ready.llm -or -not $ready.ready) {
    throw "Local readiness failed: $($ready | ConvertTo-Json -Compress)"
}

$readyAlias = Invoke-RestMethod -Method Get -Uri "$appBaseUrl/api/v1/ready"
if (-not $readyAlias.ready) { throw "Readiness alias failed" }
```

## 5. 실제 Mokpo 근거 질문

현재 hackathon index에서 관련 chunk가 검색되는 질문을 먼저 확인한다.

```powershell
$evidenceQuestion = "목포 개항 이후 도시에는 어떤 변화가 있었나요?"
$searchBody = @{
    query = $evidenceQuestion
    top_k = 3
} | ConvertTo-Json

$search = Invoke-RestMethod `
    -Method Post `
    -Uri "$appBaseUrl/api/v1/search" `
    -ContentType "application/json; charset=utf-8" `
    -Body $searchBody
if (@($search.results).Count -eq 0) {
    throw "Evidence question returned no local RAG results"
}
$search.results | Select-Object title, score, source_name
```

vLLM foreground log를 확인할 준비를 한 뒤 chat을 호출한다.

```powershell
$chatBody = @{ message = $evidenceQuestion } | ConvertTo-Json
$answer = Invoke-RestMethod `
    -Method Post `
    -Uri "$appBaseUrl/api/v1/chat" `
    -ContentType "application/json; charset=utf-8" `
    -Body $chatBody
$answerFields = @($answer.PSObject.Properties.Name)
if ($answerFields.Count -ne 1 -or $answerFields[0] -ne "answer") {
    throw "Unexpected user-facing chat response contract"
}
$answer.answer
```

이 호출에서는 vLLM access log에 새 `POST /v1/chat/completions` 한 건이 보여야 한다.
응답 JSON은 사용자-facing 계약인 `answer` 하나만 포함해야 한다.

## 6. 근거 부족 및 remote 무호출 확인

근거 부족 후보는 chat보다 `/api/v1/search`를 먼저 호출한다. 결과가 0건일 때만 해당
질문이 fallback smoke에 적합하다. 그 후 vLLM access log의 현재 마지막 요청을 확인하고
chat을 호출한다. 답변은 fallback이어야 하며 새 `/v1/chat/completions` log가 생기면
안 된다.

```powershell
$fallbackQuestion = "<현재 index에서 검색 결과가 0건인 회귀 질문>"
$fallbackSearchBody = @{
    query = $fallbackQuestion
    top_k = 3
} | ConvertTo-Json
$fallbackSearch = Invoke-RestMethod `
    -Method Post `
    -Uri "$appBaseUrl/api/v1/search" `
    -ContentType "application/json; charset=utf-8" `
    -Body $fallbackSearchBody
if (@($fallbackSearch.results).Count -ne 0) {
    throw "Question is not a valid fallback probe for the current index"
}

$fallbackBody = @{ message = $fallbackQuestion } | ConvertTo-Json
$fallback = Invoke-RestMethod `
    -Method Post `
    -Uri "$appBaseUrl/api/v1/chat" `
    -ContentType "application/json; charset=utf-8" `
    -Body $fallbackBody
if ($fallback.answer -ne "제공된 역사 자료에서 충분한 근거를 찾지 못했습니다.") {
    throw "Unexpected fallback response"
}
$fallback.answer
```

현재 `data/provisional_hackathon` index에서는
`양자컴퓨터의 큐비트 오류 정정 방법을 설명해 주세요.`가 0건을 반환하는 회귀 질문으로
검증되어 기본 fallback probe로 사용된다. 이 질문의 chat 응답은 안전 fallback이어야 하며
원격 `/v1/chat/completions` 요청이 새로 생기면 안 된다.

## 7. 자동 smoke script

tunnel, 원격 vLLM, 로컬 FastAPI를 각각 먼저 실행한 뒤 저장소 루트의 별도
PowerShell에서 실행한다. script는 SSH tunnel을 만들거나 패키지를 설치하지 않는다.

```powershell
& .\scripts\smoke_vllm_e2e.ps1
```

검사 결과는 `PASS`, `FAIL`, `SKIP`으로 출력한다. 기본 fallback 후보가 현재 index에서
검색 결과를 반환하면 실패시키지 않고 다음을 출력한다.

```text
SKIP: no zero-result fallback probe available for current index
```

다른 후보를 확인할 때는 retrieval 설정을 바꾸지 않고 parameter로만 전달한다.

```powershell
& .\scripts\smoke_vllm_e2e.ps1 `
  -FallbackProbeQuestions @("<candidate-1>", "<candidate-2>")
```

## 8. 종료

FastAPI와 SSH tunnel을 `Ctrl+C`로 종료하고 process-scoped secret을 제거한다.

```powershell
Remove-Item Env:HISTORY_LLM_API_KEY -ErrorAction SilentlyContinue
Remove-Variable llmHeaders, directBody, direct, models -ErrorAction SilentlyContinue
```

# GPU Inference Worker

학교 GPU에는 검토된 `scripts/gpu_llm_server.py`만 전달한다. Git 저장소, `.env`, 역사 자료, RAG 인덱스, 세션, 사용자 데이터, 애플리케이션 로그는 전달하지 않는다. 실제 계정, 호스트, 모델 절대경로와 토큰은 아래 placeholder 대신 대화형으로 입력하고 Git에 기록하지 않는다.

Worker의 실제 계약은 다음과 같다.

- 인증: 토큰이 설정되면 `Authorization: Bearer <TOKEN>` 필수
- `GET /health` → `{"status":"ok"}`
- `GET /ready` → `{"ready":true,"status":"ready","model":"..."}`
- `POST /v1/chat/completions`: `model`, 비어 있지 않은 `messages` 필요; `max_tokens`, `temperature`, `top_p`, `stream` 지원
- 요청 `model`은 `GPU_LLM_MODEL_ID`와 정확히 같아야 한다.
- `stream=true`는 현재 거부한다.

## A. 로컬 PC에서 worker 파일 전달

GPU 로그인 노드에서 private 디렉터리를 한 번 준비한다.

```bash
mkdir -p -m 700 gpu-worker
```

Windows PowerShell의 저장소 루트에서 worker 한 파일만 전달한다.

```powershell
scp .\scripts\gpu_llm_server.py <GPU_LOGIN_ALIAS>:gpu-worker/gpu_llm_server.py
```

이 절차는 로그인 노드와 계산 노드가 같은 home 디렉터리를 공유한다는 전제다. 공유되지 않는다면 임의 명령을 만들지 말고 학교의 공식 계산 노드 파일 전달 방식을 확인한다.

## B. 학교 GPU에서 A6000 할당 및 worker 실행

### 1. 검증된 기본 Slurm 방식

로그인 노드에서 실제 성공한 명령을 사용한다.

```bash
srun -p p02 --gres=gpu:A6000:1 --time=00:30:00 --pty bash
```

발표 시간이 30분을 넘으면 학교 정책이 허용하는 범위에서 `--time`만 조정한다. 계산 노드에 들어왔는지와 환경을 확인한다.

```bash
hostname
/usr/bin/python3 --version
nvidia-smi
/usr/bin/python3 - <<'PY'
import accelerate
import torch
import transformers

print("torch", torch.__version__)
print("transformers", transformers.__version__)
print("accelerate", accelerate.__version__)
print("cuda_available", torch.cuda.is_available())
print("gpu", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "unavailable")
PY
```

확인된 기준 환경은 Python 3.8.10, NVIDIA RTX A6000 48GB, PyTorch 2.4.1+cu121, Transformers 4.46.3, Accelerate 1.0.1이다. Worker는 표준 라이브러리 HTTP 서버와 Transformers를 사용하며 `device_map="auto"` 때문에 Accelerate가 필요하다. 이 패키지가 이미 설치된 현재 환경에서는 별도 설치가 필요 없다. vLLM과 `nvcc`는 사용하지 않으며 CUDA runtime이 포함된 PyTorch 추론에는 `nvcc`가 필요하지 않다.

`salloc --partition=p02 --gres=gpu:1` 후 `srun --pty bash`를 실행하는 방식은 가능한 대안이지만 이 환경에서 아직 검증되지 않았다. 발표용 기본 절차로 사용하지 않는다.

### 2. snapshot과 환경변수

`GPU_LLM_MODEL_PATH`에는 Hugging Face cache의 저장소 상위 디렉터리가 아니라 해당 revision의 snapshot 디렉터리를 지정한다. 즉, 그 디렉터리 바로 아래에서 `config.json`과 safetensors weight 또는 shard를 찾을 수 있어야 한다. 실제 절대경로는 문서나 history에 입력하지 않는다.

```bash
cd "$HOME/gpu-worker"

read -rsp "Local model snapshot path (<LOCAL_MODEL_SNAPSHOT_PATH>): " GPU_LLM_MODEL_PATH; echo
export GPU_LLM_MODEL_PATH

test -f "$GPU_LLM_MODEL_PATH/config.json" || { echo "config.json not found" >&2; exit 1; }
compgen -G "$GPU_LLM_MODEL_PATH/*.safetensors" >/dev/null || { echo "safetensors not found" >&2; exit 1; }

export GPU_LLM_MODEL_ID="beomi/Llama-3-Open-Ko-8B-Instruct-preview"
export GPU_LLM_HOST="127.0.0.1"
export GPU_LLM_PORT="8001"
export GPU_LLM_MAX_REQUEST_BYTES="65536"
export GPU_LLM_MAX_MESSAGES="8"
export GPU_LLM_MAX_INPUT_CHARS="12000"
export GPU_LLM_MAX_INPUT_TOKENS="6144"
export GPU_LLM_MAX_NEW_TOKENS="512"

read -rsp "Temporary GPU worker token: " GPU_LLM_AUTH_TOKEN; echo
export GPU_LLM_AUTH_TOKEN

/usr/bin/python3 gpu_llm_server.py
```

모델은 시작 시 한 번만 `AutoTokenizer`와 `AutoModelForCausalLM`으로 로드된다. `local_files_only=True`, `torch.bfloat16`, `torch.inference_mode()`를 사용한다. 정상 출력은 bind 주소와 port뿐이며 모델 경로, 토큰, prompt와 응답 원문이 출력되면 안 된다.

같은 allocation에 두 번째 shell을 여는 학교 공식 방법이 있을 때만 그 shell에 동일 토큰을 대화형으로 입력하고 다음 health/ready 검사를 실행한다. 토큰은 명령 인자로 넣지 않는다.

```bash
read -rsp "Temporary GPU worker token: " GPU_LLM_AUTH_TOKEN; echo
export GPU_LLM_AUTH_TOKEN

/usr/bin/python3 - <<'PY'
import json
import os
from urllib.request import Request, urlopen

for path in ("/health", "/ready"):
    request = Request(
        "http://127.0.0.1:8001" + path,
        headers={"Authorization": "Bearer " + os.environ["GPU_LLM_AUTH_TOKEN"]},
    )
    with urlopen(request, timeout=5) as response:
        print(path, response.status, json.loads(response.read().decode("utf-8")))
PY
```

## C. 로컬 PowerShell에서 SSH tunnel 및 앱 실행

### 1. Tunnel

`hostname`으로 확인한 계산 노드 이름은 Git에 기록하지 않는다. 계산 노드 직접 SSH가 허용될 때만 별도 PowerShell 창에서 다음 방식을 사용한다.

```powershell
ssh -N -J <GPU_LOGIN_ALIAS> -L 8001:127.0.0.1:8001 <GPU_COMPUTE_NODE>
```

계산 노드 SSH 허용 여부는 아직 검증되지 않았다. 허용되지 않으면 학교 관리자나 공식 문서에서 compute-node tunneling 방식을 확인한다. 로그인 노드의 `127.0.0.1`로 단순 포워딩하면 계산 노드의 localhost worker에는 연결되지 않는다.

Tunnel 창을 열어 둔 채 다른 PowerShell에서 확인한다.

```powershell
Test-NetConnection 127.0.0.1 -Port 8001
```

### 2. 앱 환경변수와 worker 검사

저장소 루트에서 실행한다. GPU worker에 입력한 것과 같은 토큰을 대화형으로 입력한다.

```powershell
$env:PYTHONPATH = "src"
$env:LLM_BACKEND = "openai_compatible"
$env:LLM_API_FORMAT = "openai"
$env:LLM_BASE_URL = "http://127.0.0.1:8001"
$env:LLM_MODEL = "beomi/Llama-3-Open-Ko-8B-Instruct-preview"
$env:LLM_API_KEY_REQUIRED = "true"
$env:LLM_READINESS_PROBE = "true"
$env:LLM_TIMEOUT_SECONDS = "60"
$env:LLM_MAX_NEW_TOKENS = "512"
$env:LLM_REMOTE_HISTORY_ENABLED = "false"
$env:LLM_REMOTE_SANITIZE_ENABLED = "true"
$env:LLM_API_KEY = Read-Host -MaskInput "Temporary GPU worker token"

$headers = @{ Authorization = "Bearer $env:LLM_API_KEY" }
Invoke-RestMethod -Method Get -Uri "$env:LLM_BASE_URL/health" -Headers $headers
Invoke-RestMethod -Method Get -Uri "$env:LLM_BASE_URL/ready" -Headers $headers
```

인증 누락 요청이 HTTP 401로 거부되는지 확인한다.

```powershell
try {
    Invoke-RestMethod -Method Get -Uri "$env:LLM_BASE_URL/health"
    throw "Unauthenticated request unexpectedly succeeded"
} catch {
    if ($_.Exception.Response.StatusCode.value__ -ne 401) { throw }
    "Unauthenticated request rejected with HTTP 401"
}
```

Worker에 직접 비스트리밍 한국어 요청을 보낸다.

```powershell
$directBody = @{
    model = $env:LLM_MODEL
    messages = @(
        @{ role = "system"; content = "한국어로 간단히 답하세요." }
        @{ role = "user"; content = "한 문장으로 자기소개를 해 주세요." }
    )
    max_tokens = 64
    temperature = 0.2
    top_p = 0.9
    stream = $false
} | ConvertTo-Json -Depth 5

Invoke-RestMethod `
    -Method Post `
    -Uri "$env:LLM_BASE_URL/v1/chat/completions" `
    -Headers $headers `
    -ContentType "application/json; charset=utf-8" `
    -Body $directBody
```

### 3. 기존 앱 경로

다음 CLI와 `ask` subcommand는 `history_chatbot.chat.cli`에 실제 존재한다. 첫 질문의 “붉은 등대 전시관”은 `tests/fixtures/rag/fictional_chunks.jsonl`에 있으며 검색 성공 테스트도 존재한다.

```powershell
python -m history_chatbot.chat.cli ask "붉은 등대 전시관을 알려줘"
python -m history_chatbot.chat.cli ask "달 표면의 목포 해관 분관은 언제 문을 열었나요?"
```

첫 요청은 remote-safe 최소 prompt를 worker로 보내고 citation은 로컬에서 결합한다. 두 번째 요청은 근거가 없으면 `insufficient_evidence`를 반환하고 원격 호출을 생략한다. History 전송은 기본 비활성이다. Worker는 요청 원문을 기록하지 않으므로 live 로그만으로 request body를 검사하지 말고 자동화된 remote-safe 테스트 결과와 앱 응답을 함께 확인한다.

FastAPI와 Uvicorn 선택 의존성이 이미 설치된 로컬 환경에서만 HTTP 앱을 실행할 수 있다. 이 절차 중 새로 설치하지 않는다.

```powershell
$env:APP_MODE = "development"
python -m uvicorn history_chatbot.chat.api:create_app --factory --host 127.0.0.1 --port 8000
```

별도 PowerShell에도 같은 LLM 환경변수를 설정한 후 비스트리밍 endpoint를 호출한다.

```powershell
Invoke-RestMethod -Method Get -Uri "http://127.0.0.1:8000/api/readiness"

$appBody = @{
    user_query = "붉은 등대 전시관을 알려줘"
    conversation_mode = "free_chat"
} | ConvertTo-Json

Invoke-RestMethod `
    -Method Post `
    -Uri "http://127.0.0.1:8000/api/chat" `
    -ContentType "application/json; charset=utf-8" `
    -Body $appBody
```

`APP_MODE=development`는 fictional fixture RAG를 사용한다. 검토된 실제 자료용 `create_development_real_service()`는 코드에 있지만 기본 ASGI factory나 CLI에 등록되어 있지 않다. 실제 자료 HTTP UI까지 필요하면 production 보호를 우회하지 않는 별도 로컬 entry point가 필요하다.

## D. 안전한 종료와 환경변수 정리

1. 로컬 앱을 `Ctrl+C`로 종료한다.
2. SSH tunnel 창을 `Ctrl+C`로 종료한다.
3. GPU worker를 `Ctrl+C`로 종료한다.
4. 계산 노드에서 민감 변수를 제거하고 검증된 `srun` shell을 종료한다.

   ```bash
   unset GPU_LLM_AUTH_TOKEN GPU_LLM_MODEL_PATH
   exit
   ```

5. 로그인 노드에서 학교 공식 Slurm 상태 명령으로 job 종료를 확인한다. 미검증 `salloc` 대안을 사용했다면 compute shell과 allocation shell을 각각 종료해야 한다.
6. 토큰을 입력한 모든 PowerShell 창에서 process-scoped 값을 제거한다.

   ```powershell
   Remove-Item Env:LLM_API_KEY -ErrorAction SilentlyContinue
   Remove-Item Env:LLM_BASE_URL -ErrorAction SilentlyContinue
   Remove-Item Env:LLM_MODEL -ErrorAction SilentlyContinue
   Remove-Variable headers, directBody, appBody -ErrorAction SilentlyContinue
   ```

7. Worker 복사본이 임시라면 정확한 대상이 `gpu-worker/gpu_llm_server.py`인지 다시 확인한 후 그 파일만 제거한다. 모델 cache나 공유 데이터는 삭제하지 않는다.

토큰은 Bash/PowerShell 명령 인자, shell history, 프로세스 목록, `.env`, Slurm script, 문서, 프록시 로그 또는 Git에 넣지 않는다. `Read-Host -MaskInput`을 지원하지 않는 PowerShell에서는 기관이 승인한 대화형 secret 입력 방식을 사용한다.

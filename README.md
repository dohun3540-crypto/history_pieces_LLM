# 목포 근대역사 다국어 RAG 챗봇

## 현재 자료 준비 상태

- 실제 `reviewed + allowed_for_rag=true` 승인 자료는 현재 0건입니다.
- 개발 기능은 `tests/fixtures/rag`의 **테스트용 가상 자료이며 실제 역사 사실이
  아닌 fixture**로 검증합니다.
- fixture 테스트 통과는 소프트웨어 동작 검증이며 실제 역사 서비스나 역사 정확도
  검증 완료를 의미하지 않습니다.
- production 모드는 fixture와 MockLLM을 금지하며, 승인 자료가 0건이면 준비 미완료
  오류를 표시합니다.
- 공식 자료가 추가되면 모델을 재학습하지 않고 증분 인덱싱으로 반영합니다.
- LoRA·QLoRA는 자료 추가가 아니라 답변 방식 개선이 필요할 때만 별도로 수행합니다.

자세한 내용은 [개발 모드](docs/DEVELOPMENT_MODE.md),
[증분 갱신](docs/INCREMENTAL_UPDATE_GUIDE.md),
[API 대기 상태](docs/API_PENDING_STATUS.md)를 참고하세요.

### Development 대화형 RAG 실행

실제 역사 자료나 Llama 모델 없이 가상 fixture와 MockLLM으로 전체 경로를
확인할 수 있습니다. 출력은 반드시 테스트용 응답으로 표시됩니다.

```powershell
python -m history_chatbot.chat.cli ask "붉은 등대 전시관을 알려줘"
python -m history_chatbot.chat.cli ask "그 건물은 어떤 설정이야?" --session-id "<SESSION_ID>"
python -m history_chatbot.chat.cli reset --session-id "<SESSION_ID>"
```

HTTP API는 선택 사항입니다.

```powershell
python -m pip install -e ".[api]"
uvicorn history_chatbot.chat.api:create_app --factory
```

같은 서버가 offline reference web UI도 제공한다. 로컬 loopback으로 실행한 뒤
`http://127.0.0.1:8000/`을 연다.

```powershell
$env:APP_MODE = "development"
python -m uvicorn history_chatbot.chat.api:create_app --factory --host 127.0.0.1 --port 8000
```

UI는 in-memory demo journey와 중립적인 조각 label을 사용하며 서버 재시작 시 상태가
사라진다. 실제 사진·기록새 이미지·게임 DB·저장·지도·시설 provider는 연결되어 있지
않다. [Web UI](docs/WEB_UI.md)와 [Chat API](docs/CHAT_API.md)를 참고한다. 기록새의
최종 정체성·말투·출력 영역은
[최종 캐릭터 원칙](docs/GIROKSAE_CHARACTER_PRINCIPLES_V11.md)을 유일한 기준으로 한다.

자세한 흐름과 응답 형식은 [E2E RAG](docs/END_TO_END_RAG.md),
[Chat API](docs/CHAT_API.md), [세션 관리](docs/SESSION_MANAGEMENT.md)를
참고하세요.

### Remote LLM 준비 상태

현재 실제 Llama 모델이나 외부 추론 서버는 연결하지 않았습니다. MockLLM과
가짜 transport 테스트 통과는 실제 모델 품질 검증이 아닙니다. 향후 GPU 서버의
허용된 URL과 모델명을 환경변수에 설정하면 OpenAI-compatible 또는 프로젝트 전용
FastAPI 계약으로 교체할 수 있습니다.

```powershell
$env:HISTORY_LLM_BACKEND="openai_compatible"
$env:HISTORY_LLM_BASE_URL="<허용된 GPU 서버 URL>"
$env:HISTORY_LLM_ALLOWED_HOSTS="<GPU 서버 호스트명>"
$env:HISTORY_LLM_MODEL_ID="<정확한 model ID>"
$env:HISTORY_LLM_MODEL_REVISION="<고정 revision>"
$env:LLM_READINESS_PROBE="true"
```

실제 배포 전 모델 라이선스와 context window를 확인해야 합니다. 자세한 계약은
[Remote backend](docs/REMOTE_LLM_BACKEND.md),
[서버 계약](docs/LLM_SERVER_CONTRACT.md),
[운영 readiness](docs/PRODUCTION_READINESS.md)를 참고하세요.

목포 근대역사 자료를 검색(Retrieval)하고, 검색된 근거만으로 답변을 생성하는
다국어 확장형 챗봇의 1차 프로토타입입니다. 현재는 한국어 입력을 중심으로 하며,
실제 Llama 모델 대신 `MockLLM`을 사용합니다.

> 이 저장소의 샘플 데이터는 동작 확인용 표본이며 역사 지식 자료가 아닙니다.
> 서비스에 사용하기 전에 공신력 있는 출처를 수집하고 전문가의 사실 검증을
> 거쳐야 합니다.

## 설계 원칙

- 역사 지식은 모델 가중치가 아니라 RAG 문서 계층에서 관리합니다.
- 모델은 `BaseLLM`, 검색기는 `BaseRetriever` 인터페이스 뒤에서 교체합니다.
- 대화 기억은 세션 메모리이며 모델 학습이나 학습 데이터 추가를 수행하지 않습니다.
- `original_query`를 보존하고 검색에는 NFC/공백 정리된 `normalized_query`를 사용합니다.
- 현재 단계에서는 모델, 대형 데이터셋, 체크포인트를 다운로드하지 않습니다.

## 실행

Python 3.11 이상이 필요합니다.

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
python -m pip install -e ".[dev]"
history-chatbot
```

한 번만 질문하려면:

```bash
history-chatbot "목포 샘플 자료를 알려줘"
```

개발 환경에서 설치 없이 실행할 수도 있습니다.

```powershell
$env:PYTHONPATH="src"
python -m history_chatbot.cli "목포 샘플"
```

## 테스트

```bash
pytest
```

설정 파일은 `configs/`, 검증된 원문 데이터의 작성 지침은
`docs/DATA_GUIDE.md`, 전체 구조와 향후 GPU 튜닝 계획은 `docs/`에서 확인할
수 있습니다.

## 자료 수집·정제 파이프라인

파이프라인은 인터넷 수집 없이 로컬 UTF-8 TXT/Markdown 자료를 등록하고, 저작권
정책 검증, 텍스트 정제, 청킹, JSONL 출력을 수행합니다. PDF는 선택적 `pypdf`가
설치된 경우만 텍스트 추출을 시도하며 HTML과 크롤링은 지원하지 않습니다.

PowerShell에서 저장소 루트를 현재 디렉터리로 두고 실행합니다.

```powershell
$env:PYTHONPATH = "src"
python -m history_chatbot.ingestion.cli register --manifest "data/manifests/sources.jsonl" --metadata "C:\작업\source-metadata.json"
python -m history_chatbot.ingestion.cli process --manifest "data/manifests/sources.jsonl" --document-id "mokpo-source-001"
python -m history_chatbot.ingestion.cli validate --manifest "data/manifests/sources.jsonl" --document-id "mokpo-source-001"
python -m history_chatbot.ingestion.cli list --manifest "data/manifests/sources.jsonl"
```

원문은 `data/raw` 아래에 두되 대용량 원문·PDF·데이터셋은 Git에 커밋하지 않습니다.
등록 형식과 검수 승격 절차는 [자료 처리 가이드](docs/INGESTION_GUIDE.md),
이용 정책은 [출처 정책](docs/SOURCE_POLICY.md)을 따릅니다. `reviewed`가 아닌
자료는 서비스용 RAG 색인 대상이 아닙니다.

사람 검수자는 다음 명령으로 manifest 문서를 조회·승인·거절할 수 있습니다.
manifest 기본값은 `data/manifests/sources.jsonl`, 로컬 감사 로그 기본값은
`data/manifests/review_audit.jsonl`입니다.

```powershell
$env:PYTHONPATH = "src"
python -m history_chatbot.ingestion.cli review show --document-id "mokpo-source-001"
python -m history_chatbot.ingestion.cli review approve --document-id "mokpo-source-001" --reviewer "검수자"
python -m history_chatbot.ingestion.cli review reject --document-id "mokpo-source-001" --reviewer "검수자" --reason "출처를 확인할 수 없음"
```

`unknown`·`restricted` 저작물은 승인할 수 없습니다. 검수 승인은
`allowed_for_rag`를 자동 변경하지 않으므로 이 값이 `false`이면 승인 후에도
서비스 RAG 색인 대상이 아닙니다.

## 검수 완료 RAG 입력 준비

사람 검수가 끝나고 RAG 사용이 허용된 A·B 등급 문서만 검색 입력용 JSONL로
준비합니다. 아직 임베딩 모델, 벡터 DB와 Llama 모델은 연결하지 않습니다.

```powershell
$env:PYTHONPATH = "src"
python -m history_chatbot.indexing.cli status
python -m history_chatbot.indexing.cli list-eligible
python -m history_chatbot.indexing.cli list-rejected
python -m history_chatbot.indexing.cli prepare
python -m history_chatbot.indexing.cli validate
```

결과는 `data/index_ready/chunks.jsonl`과
`data/index_ready/index_manifest.json`에 생성됩니다. 생성 데이터는 Git에서
제외하며, 문서별 스냅샷 해시·증분 변경 상태·tombstone을 manifest에 기록합니다.
자세한 정책과 출력 형식은 [인덱싱 가이드](docs/INDEXING_GUIDE.md)를 참고하세요.

## 공식 자료 후보 수집

공식 출처 seed를 이용한 제한적 후보 수집기는 robots.txt와 허용 도메인을 먼저
확인하고, 요청 지연·타임아웃·재시도를 적용합니다. API URL이 명시된 출처는 API를
우선합니다. 기본 seed에는 인증 키가 필요한 API를 임의로 등록하지 않았습니다.

```powershell
$env:PYTHONPATH = "src"

# 기본 동작은 네트워크 요청 없는 dry-run
python -m history_chatbot.collectors.cli `
  --source-id "heritage_portal" `
  --query "목포 근대역사문화공간" `
  --dry-run

# 실제 실행은 --execute를 명시해야 함
python -m history_chatbot.collectors.cli --query "목포 개항" --execute
```

다운로드된 응답 원본은 `data/raw/collected`, 추출 텍스트는
`data/extracted/collected`, 후보 목록은 `data/source_catalog/collected_sources.jsonl`
에 서로 분리됩니다. 이 산출물은 대용량·권리 미확인 자료이므로 Git에서 제외됩니다.
모든 후보는 `draft`, `copyright_status=unknown`, RAG·학습 사용 금지로 시작합니다.
`collection_status=allowed`이면서 `robots_verification=verified`인 출처만 실행되며,
출처별 최대 2건·전체 최대 10건을 코드에서 고정합니다. 일반 CLI와 개별 수집기
호출도 이 제한을 우회할 수 없습니다. 로그인·캡차·유료벽 징후가 발견되면 해당
페이지를 저장하지 않고 건너뜁니다.
운영 원칙은 [수집 정책](docs/COLLECTION_POLICY.md), 출처 목록은
[신뢰 출처 안내](docs/TRUSTED_SOURCES.md)를 확인하세요.

## 하이브리드 검색

기록새의 17개 상황·54개 human-authored seed, 선택적 RAG, 대화 모드와
개인화 후보 정책은 [기록새 상황 대화 연결](docs/GIROKSAE_DIALOGUE.md)을 참고하세요.
실제 multilingual-e5-small 설치·색인 방식은
[임베딩 backend](docs/EMBEDDING_BACKEND.md), 30개 한중 검색 실측과 제한은
[다국어 검색 평가](docs/MULTILINGUAL_SEARCH_EVALUATION.md)를 참고하세요.

8단계 `index_ready` 청크만 대상으로 dense 후보와 BM25 결과를 융합합니다.
현재 dense 기본값 `hashing-v1`은 모델 다운로드 없는 개발·테스트 구현이며,
실제 의미 임베딩 모델로 가장하지 않습니다. 임베딩 후보와 다운로드 조건은
[검색 모델 보고서](docs/RETRIEVAL_MODEL_REPORT.md), 실행·안전 정책은
[검색 가이드](docs/RETRIEVAL_GUIDE.md)를 참고하세요.

```powershell
python -m pip install -e ".[dev]"
python -m history_chatbot.retrieval.cli inspect-models
python -m history_chatbot.retrieval.cli build-index
python -m history_chatbot.retrieval.cli status
python -m history_chatbot.retrieval.cli search "목포 개항"
python -m history_chatbot.retrieval.cli benchmark
```

검수 완료 eligible 문서가 0건이면 빈 인덱스를 정상 생성하고, 모든 검색은
근거 없는 문서를 반환하는 대신 빈 결과를 낸다. `draft`·`rejected` 문서는
색인하지 않는다.

## 비상업적 해커톤 임시 모드

현재 local manifest 50건과 정제된 48문서는 정식 승인 자료가 아니다.
정제 결과는 현재 239청크이며, 모두 `allowed_for_rag=false`,
`allowed_for_training=false`를 유지한 채 비상업적 대학 산학 해커톤에서만
`provisional_hackathon`으로 격리 사용할 수 있다.

```powershell
$env:APP_MODE = "hackathon"
$env:HISTORY_LLM_BACKEND = "mock" # 로컬 연결 검증 전용, 실제 Llama 성공이 아님
python -m history_chatbot.provisional.cli dry-run
python -m history_chatbot.provisional.cli prepare
python -m history_chatbot.provisional.cli list
python -m history_chatbot.provisional.cli rebuild
python -m uvicorn history_chatbot.chat.api:create_app --factory --host 127.0.0.1 --port 8000
```

원문·정제문·해커톤 인덱스는 공개 Git에 올리지 않는다. production은 임시
자료를 자동 제외하며 탐지 시 로드 자체를 거부한다. 기관 요청, 권리 문제 또는
2026-08-31 재검토 시 source_id·기관·전체 단위로 제거할 수 있다. 자세한 내용은
[해커톤 자료 정책](docs/HACKATHON_DATA_POLICY.md),
[수명주기](docs/PROVISIONAL_DATA_LIFECYCLE.md),
[제거 안내](docs/PROVISIONAL_DATA_REMOVAL.md)를 참고한다.

`data/provisional_hackathon/processed/chunks.jsonl`은 Git에 포함되지 않으므로 새로
받은 저장소에서는 먼저 허용된 로컬 원문으로 `collect`/`reprocess-local`과 `rebuild`를
완료해야 한다. 파일이 준비된 현재 작업 환경의 readiness는 해커톤 데모용으로만
ready이며 정식 서비스 준비 완료를 의미하지 않는다. 브라우저 주소는
`http://127.0.0.1:8000/`, API 문서는 `http://127.0.0.1:8000/docs`다.

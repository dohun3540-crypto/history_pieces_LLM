# 목포 근대역사 다국어 RAG 챗봇

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

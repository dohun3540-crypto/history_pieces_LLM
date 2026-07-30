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

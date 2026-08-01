# Multilingual E5 embedding backend

## 설치 환경

- Python: 3.13.9
- 격리 환경: `C:\projects\history_pieces_LLM\.venv`
- PyTorch: 2.13.0+cpu
- sentence-transformers: 5.6.1
- transformers: 5.14.1
- huggingface-hub: 1.26.0
- 실행 device: CPU (`cuda_available=false`)
- 논리 CPU: 16
- 측정 당시 C: 여유 공간: 115.52GB

Anaconda base에 직접 설치했을 때 OpenMP runtime 중복이 탐지되어 결과 신뢰성을 위해
프로젝트 `.venv`를 생성했다. `.venv`는 Git 제외 대상이다. 프로젝트 의존성은
`pip install -e ".[embedding]"`으로 재현한다.

## 모델과 캐시

- 모델: `intfloat/multilingual-e5-small`
- 고정 revision: `614241f622f53c4eeff9890bdc4f31cfecc418b3`
- dimension: 384
- 최대 입력 길이: 512 token
- query prefix: `query: `
- passage prefix: `passage: `
- `normalize_embeddings=true`; 실측 L2 norm은 1.0
- 캐시: `.runtime/model_cache/huggingface`
- 캐시 크기: 493,385,465 bytes (470.53MiB), 14개 물리 파일

모델 저장소 전체는 ONNX, OpenVINO, PyTorch 중복 가중치를 포함해 약 2.17GiB다.
이번 작업에서는 safetensors, tokenizer와 설정 등 실행 필수 10개 snapshot 파일만
선택 다운로드했다. 모델 파일은 `.runtime` 아래에 있어 Git에 포함되지 않는다.

## 인덱스

기존 hashing 인덱스는 다음 위치에 그대로 유지한다.

` .runtime/indexes/hackathon/hashing-v1--builtin.json `

E5 인덱스는 다음 별도 위치에 생성한다.

` .runtime/indexes/hackathon/e5/intfloat--multilingual-e5-small--614241f622f53c4eeff9890bdc4f31cfecc418b3.json `

실측 결과:

- 48 documents / 133 chunks / 133 vectors
- 384 dimensions
- NaN/Inf 0, 빈 벡터 0, 중복 chunk ID 0, source ID 누락 0
- lane: `provisional_hackathon`
- 모델 로드: 4.0078초
- passage embedding: 17.6951초
- 청크당 평균: 133.0455ms
- 전체 build: 17.8062초
- 저장 시간 추정: 0.1112초
- 인덱스 크기: 1,964,474 bytes

metadata에는 schema/format/index version, backend, model/revision, dimension,
normalization, 두 prefix, 생성 시각, corpus fingerprint/source snapshot, 청크·문서 수,
data lane을 기록한다. 모델, revision, dimension 또는 normalization이 다르면 검색을
거부한다. 자동 hashing fallback은 없고 hashing은 명시적 설정에서만 선택한다.

후보 설정은 `configs/retrieval.e5.candidate.yaml`에 있으며 production 기본 설정을
변경하지 않는다. 실제 결과와 제한은 [다국어 검색 평가](MULTILINGUAL_SEARCH_EVALUATION.md)를
참고한다.

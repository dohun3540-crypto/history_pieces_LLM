# 하이브리드 검색 가이드

## 안전 경계

검색 인덱스는 8단계가 만든 다음 두 파일만 입력으로 사용한다.

- `data/index_ready/chunks.jsonl`
- `data/index_ready/index_manifest.json`

원문, 수집 후보, ingestion manifest를 검색기가 직접 읽지 않는다.
`chunks.jsonl` 해시가 manifest와 다르거나 활성 문서가 아니거나 검수 이력이
없으면 빌드를 거부한다. tombstone 문서는 새 인덱스에서 제거된다.

## 현재 백엔드

기본값은 표준 라이브러리만 사용하는 `local_json`이다. 개발 자료가 적고,
네트워크 없이 파일 하나를 원자적으로 교체·검증할 수 있어 모델 승인 전
Qdrant Local보다 실패 면적이 작다. 애플리케이션은 `VectorStore` 인터페이스를
사용하므로 운영 시 Qdrant 서버 어댑터로 교체할 수 있다.

`QdrantVectorStore`는 선택적 확장 지점만 제공한다. `qdrant-client`가 없으면
명확한 오류를 내며, 미구현 상태에서 조용히 다른 저장소로 대체하지 않는다.

## 검색 흐름

1. 한국어 NFC·공백 정규화
2. dense 후보 검색
3. BM25 후보 검색
4. reciprocal-rank fusion
5. 선택적 reranker
6. 중복 제거 및 문서별 최대 청크 제한
7. 점수와 근거 임계값 적용
8. 부족하면 빈 결과

`목포`처럼 모든 자료에 흔한 단어만 겹칠 때는 결과를 반환하지 않는다.
질의의 구체적인 토큰이 문서에 있거나 충분히 높은 dense 점수가 있어야 한다.
실제 임베딩 모델을 연결할 때는 benchmark로 임계값을 다시 보정해야 한다.

## PowerShell 실행

개발 설치 후:

```powershell
python -m pip install -e ".[dev]"
python -m history_chatbot.indexing.cli prepare
python -m history_chatbot.retrieval.cli inspect-models
python -m history_chatbot.retrieval.cli build-index
python -m history_chatbot.retrieval.cli status
python -m history_chatbot.retrieval.cli search "목포 개항"
python -m history_chatbot.retrieval.cli benchmark
```

강제 재색인은 다음과 같다.

```powershell
python -m history_chatbot.retrieval.cli build-index --rebuild
```

현재 eligible 문서가 0건이어도 `build-index`는 정상적으로 빈 오프라인
인덱스를 만들고 명확한 안내를 출력한다.

## 증분·삭제·버전 관리

- 동일 `chunk_id`와 `content_sha256`는 기존 벡터를 재사용한다.
- 변경되거나 새로 생긴 청크만 다시 인코딩한다.
- 현재 `index_ready`에서 사라진 청크와 tombstone 문서는 교체 저장 시 제거한다.
- 인덱스 파일명에 모델 ID와 revision을 포함한다.
- 저장된 모델 ID/revision 또는 source snapshot이 현재 설정과 다르면 검색하지
  않고 재색인을 요구한다.

## 운영 전 필수 검증

- 실제 임베딩 모델 승인과 라이선스 기록
- 한국어 동의 표현, 띄어쓰기·오타 benchmark
- 관련 질문 recall과 무관 질문 false-positive 측정
- 임계값 조정 이력 기록
- Qdrant를 선택할 경우 Local/Server 양쪽의 증분·삭제 통합 테스트

# 검색 모델 후보 조사

- 조사일: 2026-07-30
- 조사 범위: 한국어·다국어 역사 RAG용 임베딩 및 선택적 reranker
- 현재 상태: 모델·가중치·추가 Python 패키지를 다운로드하지 않음

## 결론

첫 실제 임베딩 후보는 `intfloat/multilingual-e5-small`이다. MIT 라이선스이고
94개 언어를 대상으로 하며 384차원, 최대 512토큰 모델이다. 공식 safetensors
가중치는 약 471MB다. 현재 RTX 4060 Laptop 8GB에서 가장 낮은 위험으로
한국어 검색 품질을 시험할 수 있다.

선택적 reranker 후보는 `BAAI/bge-reranker-v2-m3`이다. Apache-2.0
라이선스의 다국어 cross-encoder지만 가중치가 약 2.27GB이므로, 임베딩
기준선이 충분하지 않다는 측정 결과가 나온 뒤에만 추가하는 편이 안전하다.

## 후보 비교

| 역할 | 정확한 모델 ID | 라이선스 | 모델/저장소 크기 | 문맥·차원 | 예상 메모리와 판단 |
|---|---|---|---|---|---|
| 추천 임베딩 | `intfloat/multilingual-e5-small` | MIT | safetensors 약 471MB, 전체 저장소 2.28GB | 512토큰, 384차원 | RAM 약 1.5~2.5GB, VRAM 약 1~2GB |
| 고성능 대안 | `BAAI/bge-m3` | MIT | 주 가중치 약 2.27GB, 전체 저장소 4.59GB | 8,192토큰, 1,024차원 | RAM 약 4~7GB, VRAM 약 3~6GB |
| 선택 reranker | `BAAI/bge-reranker-v2-m3` | Apache-2.0 | safetensors 약 2.27GB, 전체 2.29GB | 다국어 cross-encoder | RAM 약 4~7GB, VRAM 약 3~6GB |

메모리는 배치, 문맥 길이, 정밀도에 따라 달라지는 계획값이며 설치 후
실측해야 한다. 전체 저장소 크기에는 PyTorch, ONNX, OpenVINO처럼 같은 모델의
여러 포맷이 함께 포함될 수 있으므로 `snapshot_download`에서 필요한 파일만
받는 정책이 필요하다.

공식 근거:

- [`intfloat/multilingual-e5-small` 모델 카드](https://huggingface.co/intfloat/multilingual-e5-small)
- [`intfloat/multilingual-e5-small` 파일 목록](https://huggingface.co/intfloat/multilingual-e5-small/tree/main)
- [`BAAI/bge-m3` 모델 카드](https://huggingface.co/BAAI/bge-m3)
- [`BAAI/bge-m3` 파일 목록](https://huggingface.co/BAAI/bge-m3/tree/main)
- [`BAAI/bge-reranker-v2-m3` 파일 목록](https://huggingface.co/BAAI/bge-reranker-v2-m3/tree/main)

## 모델별 주의점

### multilingual-e5-small

- 비대칭 검색에서는 질의에 `query:`, 문서에 `passage:` 접두사를 사용해야
  한다고 모델 카드가 명시한다.
- 유사도 값이 높은 구간에 몰리는 특성이 있으므로 현재 개발용 해싱 인코더의
  임계값을 그대로 재사용하면 안 된다.
- 512토큰보다 긴 청크는 잘리므로 현재 `index_ready` 청크 길이와 함께
  benchmark해야 한다.

### BGE-M3

- dense, learned sparse, multi-vector 검색을 한 모델에서 제공하고 100개 이상
  언어와 최대 8,192토큰을 표방한다.
- 모델과 1,024차원 벡터가 커서 소규모 파일럿에는 비용이 높다.
- 현재 구현의 BM25와 결합해 비교한 뒤 BGE learned sparse 사용 여부를
  별도로 결정해야 한다.

### BGE reranker v2 M3

- 모든 후보 쌍을 cross-encoder로 다시 계산하므로 응답 지연이 증가한다.
- 초기에는 `NoOpReranker`를 사용하며 benchmark에서 오탐 감소 효과가 확인될
  때만 활성화한다.

## 다운로드 승인 전 표시할 정보

추천 모델 다운로드가 필요해지면 다음 조건으로 다시 승인을 요청한다.

| 항목 | 값 |
|---|---|
| 모델 ID | `intfloat/multilingual-e5-small` |
| 라이선스 | MIT |
| 인증 | 공개 저장소, 일반적으로 별도 gated 승인 불필요 |
| 필수 가중치 | `model.safetensors`, 약 471MB |
| tokenizer 포함 예상 다운로드 | 약 0.50GB |
| 안전한 디스크 예산 | 1~2GB |
| 예상 RAM | 약 1.5~2.5GB |
| 예상 VRAM | 약 1~2GB |
| 제안 저장 위치 | `models/embeddings/intfloat--multilingual-e5-small/<revision>/` |

현재 구현은 모델이 설치된 것처럼 가장하지 않는다. `hashing-v1`은 신경망
의미 모델이 아닌 다운로드 없는 개발·테스트 인코더로 명시되며, 실제 모델 ID를
설정하면 승인·설치 전에는 설정 검증 오류를 반환한다.

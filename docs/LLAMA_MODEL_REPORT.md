# 공식 Meta Llama Instruct 모델 후보 조사

- 조사일: 2026-07-30
- 범위: 공식 Meta Llama 텍스트 Instruct 모델 최대 3개
- 원칙: 모델 다운로드와 인증은 수행하지 않고 공식 모델 카드와 저장소
  파일만 조사

## 결론

추천 모델은 `meta-llama/Llama-3.2-3B-Instruct`, 대체 모델은
`meta-llama/Llama-3.2-1B-Instruct`이다.

다만 두 모델 모두 공식 지원 언어 목록에 한국어가 없다. Meta 모델 카드는
영어, 독일어, 프랑스어, 이탈리아어, 포르투갈어, 힌디어, 스페인어, 태국어를
지원 언어로 명시한다. 따라서 “한국어에 공식적으로 최적화된 모델”로
간주해서는 안 되며, 승인 후 한국어 질문과 연속 대화 품질을 반드시 실측해야
한다. RAG 근거를 제공해도 자연스러운 한국어 생성 품질은 별도 검증 대상이다.

## 후보 비교

| 후보 | 파라미터 / 공식 정밀도 | 공식 문맥 길이 | 공식 가중치 크기 | 이 하드웨어 판단 |
|---|---|---:|---:|---|
| `meta-llama/Llama-3.2-3B-Instruct` | 3.21B / BF16 | 128K | 약 6.43 GB | 4비트 GPU 우선, BF16 GPU는 매우 빠듯 |
| `meta-llama/Llama-3.2-1B-Instruct` | 1.23B / BF16 | 128K | 약 2.47 GB | BF16 GPU 가능성이 높고 CPU도 현실적 |
| `meta-llama/Llama-3.1-8B-Instruct` | 8B / BF16 | 128K | 약 16.1 GB | BF16 GPU 불가, 4비트·오프로딩도 8 GiB에서 위험 |

파일 크기는 공식 Hugging Face 저장소에 표시된 safetensors 합계를 기준으로
한다. 다운로드 과정의 캐시, 임시 파일, tokenizer와 런타임 공간은 포함하지
않는다.

## 공통 라이선스와 다운로드 조건

- Llama 3.2 후보는 **Llama 3.2 Community License Agreement**와
  Acceptable Use Policy의 적용을 받는다.
- Llama 3.1 8B 후보는 **Llama 3.1 Community License Agreement**와
  Acceptable Use Policy의 적용을 받는다.
- 세 공식 Hugging Face 저장소는 gated 모델이다. Hugging Face 로그인,
  Meta 라이선스 동의 및 접근 승인이 필요하며 다운로드 시 인증 토큰이
  필요할 수 있다.
- 토큰은 저장소, YAML, 명령 기록에 저장하지 않는다. 승인 후에도 Hugging
  Face CLI의 사용자 자격 증명 저장소나 세션 환경을 사용한다.
- 이들은 허용적 오픈소스 라이선스로 단순 분류할 수 없는 Meta의 커뮤니티
  라이선스다. 배포 및 서비스 전에 라이선스 고지, Acceptable Use Policy,
  파생 모델 명명·표시 의무를 다시 검토해야 한다.

공식 근거:

- [Llama 3.2 3B Instruct 모델 카드](https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct)
- [Llama 3.2 3B 공식 파일 목록](https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct/tree/main)
- [Llama 3.2 1B Instruct 모델 카드](https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct)
- [Llama 3.2 1B 공식 파일 목록](https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct/tree/main)
- [Llama 3.1 8B Instruct 모델 카드](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct)
- [Llama 3.1 공식 라이선스](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct/blob/main/LICENSE)

## 후보 1: Llama 3.2 3B Instruct — 추천

- 정확한 모델 ID: `meta-llama/Llama-3.2-3B-Instruct`
- 용도: 다국어 대화, 에이전트형 검색 및 요약에 맞춘 Instruct 모델
- 라이선스: Llama 3.2 Community License
- gated 여부: 예
- 인증: Hugging Face 로그인, 라이선스 동의, 접근 승인 및 다운로드 인증 필요
- Transformers: 공식 모델 카드는 `transformers >= 4.43.0` 사용 예를 제공
- chat template: 공식 tokenizer의 chat template 사용 가능
- `trust_remote_code`: 필요하지 않으며 기본값 `false` 유지
- 양자화: Meta 모델 카드는 공식 4비트 가중치/8비트 활성값 변형과 관련
  양자화 방법을 설명한다. 일반 Transformers 4비트 로딩은 별도 라이브러리의
  현재 Windows 호환성을 확인해야 한다.

다운로드 및 메모리 계획:

| 항목 | 예상 |
|---|---|
| 공식 BF16 가중치 다운로드 | 약 6.43 GB |
| 권장 여유 디스크 | 10~14 GB |
| CPU BF16 RAM | 대략 8~12 GB 이상 + 문맥 메모리 |
| GPU BF16 VRAM | 대략 7~9 GB 이상, 8 GiB에서는 OOM 위험 |
| GPU 4비트 VRAM | 짧은 문맥에서 대략 3~5 GB |
| 권장 초기 문맥 | 입력 2,048~4,096, 출력 256~512 토큰 |

추천 이유는 1B보다 답변 구성력과 지시 이행 능력을 기대할 수 있으면서 8B보다
현재 8 GiB GPU에 맞추기 쉽기 때문이다. 가장 큰 위험은 한국어가 공식 지원
언어가 아니라는 점과 Windows의 4비트 런타임 호환성이다.

## 후보 2: Llama 3.2 1B Instruct — 대체

- 정확한 모델 ID: `meta-llama/Llama-3.2-1B-Instruct`
- 라이선스: Llama 3.2 Community License
- gated 여부 및 인증: 3B와 동일
- 공식 BF16 가중치: 약 2.47 GB
- 권장 여유 디스크: 5~7 GB
- 예상 RAM: 대략 4~6 GB 이상 + 문맥 메모리
- 예상 VRAM: BF16 짧은 문맥에서 대략 3~5 GB
- Transformers와 chat template: 3B와 동일한 표준 Llama 지원 경로
- `trust_remote_code`: 필요하지 않음

3B가 VRAM, 패키지 호환성 또는 속도 요구를 충족하지 못할 때의 대체안이다.
다운로드가 작고 BF16 실행 가능성이 높지만, 복잡한 역사 질의의 한국어 표현과
긴 근거 종합 품질은 3B보다 낮을 위험이 있다. 사용자 승인 없이 실패한 3B를
이 모델로 자동 교체하지 않는다.

## 후보 3: Llama 3.1 8B Instruct — 비교용

- 정확한 모델 ID: `meta-llama/Llama-3.1-8B-Instruct`
- 라이선스: Llama 3.1 Community License
- gated 여부: 예
- 공식 BF16 가중치: 약 16.1 GB
- 권장 여유 디스크: 24~32 GB
- 예상 CPU RAM: 대략 20~28 GB + 문맥 메모리
- GPU BF16: 8 GiB VRAM에서는 불가능
- GPU 4비트: 가중치만은 접근 가능하더라도 KV 캐시와 런타임 오버헤드 때문에
  오프로딩이나 매우 짧은 문맥이 필요할 가능성이 높음

품질 여지는 크지만 이 노트북에서 안정적인 기본 백엔드로 삼기에는 메모리와
응답 지연 위험이 크므로 이번 추천에서 제외한다.

## 한국어 및 다국어 적합성

세 후보 모두 한국어를 공식 지원 언어로 명시하지 않는다. 모델 카드가
사전학습 데이터에 더 넓은 언어가 포함됐다고 설명하더라도, 이를 한국어 품질
보증으로 해석하지 않는다.

승인 후 다음 항목을 반드시 통과해야 실제 기본 백엔드로 채택한다.

1. 자연스러운 한국어 인사 생성
2. 목포 근대역사 RAG 근거를 이용한 한국어 질문 3개
3. 문맥을 유지하는 연속 대화 3턴
4. 한글 스트리밍 중 UTF-8 문자가 깨지지 않는지 확인
5. 자료에 없는 질문에서 fallback 정책을 침범하지 않는지 확인
6. 초당 토큰 수, 첫 토큰 지연, 최대 RAM/VRAM 측정

## 승인 시 적용할 다운로드 경계

추천 모델을 승인할 경우 다운로드 전에 최종적으로 다시 표시할 값은 다음과
같다.

| 항목 | 추천 모델 값 |
|---|---|
| 모델 ID | `meta-llama/Llama-3.2-3B-Instruct` |
| 라이선스 | Llama 3.2 Community License |
| 인증 | Hugging Face 로그인·라이선스 동의·접근 승인 필요 |
| 공식 가중치 다운로드 | 약 6.43 GB |
| 예상 총 디스크 예산 | 10~14 GB |
| 예상 RAM | BF16 CPU 기준 대략 8~12 GB 이상 |
| 예상 VRAM | BF16은 8 GiB에서 위험, 4비트는 대략 3~5 GB |
| 제안 저장 위치 | `models/llama/meta-llama--Llama-3.2-3B-Instruct/<revision>/` |

revision은 재현성을 위해 다운로드 시점의 커밋 해시로 고정한다. 모델 다운로드와
로딩은 별도 명령으로 유지하고, 모델이 없으면 MockLLM으로 조용히 대체하지
않고 명확한 오류를 반환하도록 구현할 예정이다.

## 현재 중단 지점

아직 모델, tokenizer, PyTorch, Transformers 또는 양자화 패키지를 다운로드하지
않았다. 백엔드 코드도 변경하지 않았다. 추천 모델과 실행 전략에 대한 사용자
승인을 받은 뒤에만 설치 호환성을 재확인하고 구현 및 다운로드 단계로
진행한다.

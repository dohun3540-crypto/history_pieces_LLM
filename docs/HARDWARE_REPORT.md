# Llama 로컬 추론 하드웨어 조사

- 조사일: 2026-07-30
- 조사 범위: 모델을 내려받거나 새 패키지를 설치하지 않은 현재 로컬 환경
- 저장소: `C:\projects\history_pieces_LLM`

## 요약

이 시스템은 NVIDIA GeForce RTX 4060 Laptop GPU(전용 VRAM 8,188 MiB)와
31.31 GiB 시스템 RAM을 갖추고 있다. Llama 3.2 1B 모델은 비교적 여유 있게,
3B 모델은 4비트 양자화와 보수적인 문맥 길이를 사용할 때 현실적으로 실행할
수 있을 것으로 예상한다. 3B BF16 전체를 GPU에 올리는 방식은 KV 캐시와 생성
중간 메모리까지 고려하면 VRAM이 빠듯하다.

현재 `torch`, `transformers` 등 추론 패키지는 설치되어 있지 않다. NVIDIA
드라이버는 CUDA 12.5 호환성을 표시하지만, PyTorch CUDA가 실제로 동작하는지는
프레임워크 설치 전에는 확인할 수 없다.

## 조사 결과

| 항목 | 확인 결과 |
|---|---|
| 운영체제 | Windows 11 Home, 64비트 |
| Windows 버전 | 10.0.26200, 빌드 26200 |
| Python | 3.13.9, 64비트 |
| Python 실행 파일 | Anaconda 환경의 `python.exe` |
| CPU | AMD Ryzen 7 8845HS with Radeon 780M Graphics |
| CPU 코어 | 물리 8코어, 논리 16코어 |
| 시스템 RAM | 33,618,251,776 bytes (31.31 GiB) |
| NVIDIA GPU | NVIDIA GeForce RTX 4060 Laptop GPU |
| 전용 VRAM | 8,188 MiB |
| NVIDIA 드라이버 | 556.29 |
| 드라이버 표시 CUDA 버전 | 12.5 |
| 내장 GPU | AMD Radeon 780M Graphics |
| C: 남은 공간 | 121.57 GiB |
| C: 사용 공간 | 354.10 GiB |

VRAM은 Windows CIM의 `AdapterRAM` 값이 아니라 `nvidia-smi`의 8,188 MiB를
기준으로 했다. CIM의 해당 필드는 큰 VRAM 값을 부정확하게 표현할 수 있다.

## Python 추론 패키지 상태

| 패키지 | 상태 |
|---|---|
| `torch` | 미설치 |
| `transformers` | 미설치 |
| `accelerate` | 미설치 |
| `bitsandbytes` | 미설치 |
| `llama-cpp-python` | 미설치 |

Python 3.13에서는 특히 CUDA용 PyTorch, Windows용 `bitsandbytes`,
`llama-cpp-python` 휠의 조합을 설치 전에 다시 확인해야 한다. 호환되는 휠이
없으면 이 저장소 전용 Python 3.11 또는 3.12 가상환경이 더 안전한 선택이다.
이번 조사에서는 환경을 만들거나 변경하지 않았다.

## 실행 방식별 판단

### Transformers GPU

- Llama 3.2 1B BF16은 8 GiB VRAM에서 중간 길이 문맥으로 실행 가능성이 높다.
- Llama 3.2 3B BF16 가중치는 약 6.43 GB이므로, KV 캐시·활성값·CUDA
  오버헤드를 포함하면 8 GiB VRAM에서는 매우 빠듯하거나 메모리 부족이 날 수
  있다.
- Llama 3.2 3B 4비트는 짧거나 중간 길이 문맥에서 현실적인 우선안이다.
- 모델이 지원하는 최대 128K 문맥 전체를 이 하드웨어에서 사용하는 것은
  현실적이지 않다. 초기 애플리케이션 한도는 입력 2,048~4,096 토큰, 출력
  256~512 토큰을 권장한다.

### CPU 및 로컬 경량 백엔드

- 31.31 GiB RAM으로 1B/3B 모델의 CPU 로딩은 가능할 것으로 보인다.
- GGUF 양자화와 `llama.cpp` 계열 백엔드는 제한된 VRAM의 대안이지만, 공식
  Meta Hugging Face 저장소는 일반적으로 GGUF를 직접 제공하지 않는다.
- 출처가 불분명한 제3자 GGUF를 자동으로 사용하지 않고, 승인받은 공식
  가중치를 내려받은 뒤 필요하면 로컬 변환하는 방식이 모델 출처 추적에 더
  안전하다.

## 성능 예상과 확인 필요 사항

아래 수치는 실측이 아니라 비슷한 사양을 바탕으로 한 계획 범위이며, 전력
제한과 프롬프트 길이에 따라 크게 달라진다.

| 구성 | 예상 생성 속도 | 주요 위험 |
|---|---:|---|
| 3B, GPU 4비트, 짧은 문맥 | 약 20~50 token/s | Windows 양자화 패키지 호환성 |
| 3B, GPU BF16 | 실행 불가~약 20~35 token/s | 8 GiB VRAM 부족 가능성 |
| 3B, CPU 양자화 | 약 8~20 token/s | 긴 문맥에서 지연 증가 |
| 1B, GPU BF16 | 약 35~80 token/s | 모델 품질과 한국어 응답력 |

승인 후 실제 백엔드를 설치할 때 다음을 다시 검증해야 한다.

1. 선택한 Python 버전용 PyTorch CUDA 휠
2. `torch.cuda.is_available()`과 실제 GPU 할당
3. 선택한 양자화 라이브러리의 Windows 지원
4. 짧은 한국어 입력에서 VRAM/RAM 사용량 및 생성 속도
5. 노트북 전원 모드와 GPU 전력 제한에 따른 편차

## 조사 방법

- Windows와 하드웨어: PowerShell CIM 조회
- GPU와 VRAM: `nvidia-smi`
- Python: `python --version`, 인터프리터·플랫폼 조회
- 패키지: Python import 명세 및 설치 패키지 조회
- 디스크: PowerShell 드라이브 조회

이번 단계에서는 모델 파일 다운로드, Hugging Face 인증, CUDA 패키지 설치,
환경 변경을 수행하지 않았다.

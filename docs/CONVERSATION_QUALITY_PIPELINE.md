# 대화 자연스러움 중심 학습·평가 파이프라인

## 품질 순서와 실행 경계

평가는 근거성, 문맥 유지, 자연스러운 fallback, 직접성, 완결성, 일반화 순서로
판단한다. `scripts/evaluate_conversation_quality.py`는 실제
`HistoryChatService`와 동일한 orchestrator/retrieval 경로를 사용한다. mock backend
결과는 retrieval·context·fallback 계약만 검증하며 생성 품질 점수를 만들지 않는다.
직접성, claim-level groundedness, 자연스러움, 전체 대화 품질은 원격 모델이 실제로
연결된 경우에도 review queue로 남긴다.

```powershell
.\.venv\Scripts\python.exe scripts/evaluate_conversation_quality.py --split all --backend mock --output .runtime/conversation-after.json
```

## 데이터와 leakage

`evaluation/conversation_quality`의 10 scenarios / 26 turns는 평가 전용이다.
scenario/topic/document 단위 split을 사용하며 random turn split을 하지 않는다.
holdout의 expected answer와 behavior는 prompt, few-shot, threshold 선택에 사용하지
않는다. `validate_splits()`는 scenario ID, topic group, document ID가 여러 split에
나타나면 실패한다.

## 지표와 실패 분류

자동 지표는 evidence/source 계약, contextualized query 일치, 문장 완결성, 동일
fallback 반복, out-of-scope 무근거 검색 여부다. 결과는 다음 원인 코드로 분류한다.

- `RETRIEVAL_FAILURE`: answerable turn인데 근거가 없음
- `CONTEXTUALIZATION_FAILURE`: 기대한 entity/topic query를 만들지 못함
- `CONTEXT_MEMORY_FAILURE`: 실제 retrieval trace를 이어 쓰지 못함
- `FALLBACK_FAILURE`: 반복되거나 유용하지 않은 fallback
- `GENERATION_FAILURE`: 모델 호출/출력 실패
- `TRUNCATION_FAILURE`: 열린 괄호 또는 미완성 꼬리
- `HALLUCINATION`: evidence/source 계약 위반 또는 review에서 근거 밖 claim
- `TOPIC_SWITCH_FAILURE`: 명시된 새 topic에 이전 topic이 섞임
- `EVIDENCE_GAP`: corpus 자체에 답이 없음
- `OUT_OF_SCOPE_FAILURE`: 범위 밖 질문에 관련 없는 문서를 붙임
- `OTHER`: 위 분류로 설명되지 않음

낮은 점수만을 이유로 threshold를 낮추지 않는다. retrieval, contextualization,
session evidence memory, prompt, generation setting을 분리해 원인을 기록한다.

## SFT/LoRA 권리 gate

현재 tracked development corpus와 local verified hackathon corpus는 모두
`allowed_for_training=false` 또는 “Not approved for training”으로 표시되어 있다.
따라서 이 저장소에는 역사 factual SFT 정답 JSONL을 생성하지 않았다. 평가 데이터를
학습 파일로 이름만 바꾸는 것도 금지한다.

권리 검수가 끝난 별도 GPU 서버에서는 한 sample을 다음 JSONL 구조로 준비한다.

```json
{"sample_id":"...","training_eligible":true,"source_evidence":[{"document_id":"...","chunk_ids":["..."],"allowed_for_training":true}],"messages":[{"role":"system","content":"목포 근대역사 안내 역할과 evidence grounding 정책"},{"role":"user","content":"[검증 근거]\n...\n[이전 대화]\n...\n[현재 질문]\n..."},{"role":"assistant","content":"근거 안의 자연스럽고 완결된 모범답변"}]}
```

train과 validation은 entity/topic 단위로 분리하고 holdout 파일은 training 서버에
복사하지 않는다. `scripts/train_lora.py`는 local checkpoint/tokenizer, CUDA PyTorch,
Transformers/PEFT/TRL/Datasets, 모든 source의 training permission을 확인한 뒤에만
`--run`을 허용한다.

```bash
python scripts/train_lora.py --config configs/lora_sft.example.yaml
python scripts/train_lora.py --config configs/lora_sft.example.yaml --run
```

첫 명령은 preflight만 수행한다. 두 번째 명령만 adapter를 별도 `output_dir`에 쓰며
원본 checkpoint를 변경하지 않는다. seed, epoch, batch, accumulation, learning rate,
LoRA rank/alpha/dropout, validation loss와 best checkpoint가 기록된다. train loss만으로
성공을 선언하지 않으며 학습 뒤 holdout 대화 평가는 serving process와 분리해 실행한다.

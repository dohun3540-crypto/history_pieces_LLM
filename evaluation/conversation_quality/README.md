# 목포 역사 대화 품질 데이터

이 디렉터리의 JSONL은 모델 학습용이 아니라 **평가 및 prompt 개발 전용**이다.
현재 근거 corpus는 `allowed_for_training=false`이므로 모든 scenario에
`training_eligible=false`를 명시한다. 이 데이터를 SFT/LoRA 입력으로 복사하거나
holdout 정답을 few-shot에 사용하면 안 된다.

split은 scenario와 주제 묶음 단위다. `train_dev`는 목포역·동양척식주식회사,
`validation`은 구 목포 일본영사관, `holdout_test`는 목포진을 중심으로 분리했다.
범위 밖 질문은 역사 주제와 별도의 `safety:*` topic group으로 관리한다. turn 단위
random split은 금지하며 `validate_splits()`가 scenario ID, topic group, factual
evidence ID의 split 간 중복을 거부한다.

각 line은 `scenario_id`, `split`, `topic_group`, `source_evidence`,
`conversation_turns`, `training_eligible`, `notes`를 가진다. 각 turn은
`user_message`, `expected_answer` 또는 `expected_behavior`, 필요 시
`expected_contextualized_query`, `answerability`, `evaluation_tags`,
`hallucination_risk`, `notes`를 가진다.

현재 구성은 10 scenarios / 26 user turns다.

- train/dev: 4 scenarios / 11 turns
- validation: 3 scenarios / 7 turns
- holdout test: 3 scenarios / 8 turns

"""향후 실제 생성 모델에 전달할 프롬프트 템플릿."""

SYSTEM_PROMPT_KO = """당신은 목포 근대역사 안내 도우미입니다.
반드시 제공된 문맥만 사용하고, 근거가 없으면 정해진 fallback 문구로 답하세요.
답변과 함께 자료의 title과 source를 표시하세요."""


def build_rag_prompt(query: str, contexts: list[str]) -> str:
    joined = "\n\n".join(contexts)
    return f"{SYSTEM_PROMPT_KO}\n\n[문맥]\n{joined}\n\n[질문]\n{query}"

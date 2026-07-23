from typing import Any


FALLBACK_CHAT_MODELS = (
    "Qwen/Qwen2.5-3B-Instruct",
    "meta-llama/Llama-3.2-3B-Instruct",
    "google/gemma-2-2b-it",
)


def get_chat_model_candidates(model: str | None = None) -> list[str]:
    candidates: list[str] = []
    if model:
        candidates.append(model)
    for fallback_model in FALLBACK_CHAT_MODELS:
        if fallback_model not in candidates:
            candidates.append(fallback_model)
    return candidates


def is_unsupported_model_error(error: Exception) -> bool:
    message = str(error).lower()
    return (
        "not supported" in message
        or "unsupported" in message
        or ("model" in message and "supported" in message)
        or "provider" in message and "enabled" in message
    )


def build_fallback_answer(question: str, sources: list[dict[str, str]]) -> str:
    if not sources:
        return (
            "현재 문맥이 충분하지 않아 정확한 요약을 생성할 수 없습니다. "
            "뉴스나 SEC 공시 데이터가 아직 준비되지 않았거나, 현재 AI 서비스가 연결되지 않았습니다."
        )

    highlights = []
    for source in sources[:3]:
        text = (source.get("text") or "").strip().replace("\n", " ")
        if text:
            highlights.append(text[:220])

    if not highlights:
        return (
            f"질문 '{question}'에 대해 현재 문맥을 바탕으로 직접 확인할 수 있는 핵심 정보가 없습니다."
        )

    joined = "\n- ".join(highlights)
    return (
        "현재 AI 모델 연결이 불안정하여 직접 요약을 생성하지 못했습니다. "
        "문맥의 핵심은 다음과 같습니다.\n"
        f"- {joined}\n\n"
        f"질문 '{question}'에 대해서는 위 문맥을 바탕으로 직접 재확인해 주세요."
    )


def run_chat_completion(
    client: Any,
    *,
    model: str | None,
    messages: list[dict[str, str]],
    max_tokens: int = 1024,
    temperature: float = 0.1,
) -> tuple[Any, str]:
    last_error: Exception | None = None
    for candidate_model in get_chat_model_candidates(model):
        try:
            response = client.chat_completion(
                model=candidate_model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
            )
            return response, candidate_model
        except Exception as error:  # pragma: no cover - exercised via tests
            if not is_unsupported_model_error(error):
                raise
            last_error = error
    if last_error is not None:
        raise last_error
    raise RuntimeError("No chat model candidates available")

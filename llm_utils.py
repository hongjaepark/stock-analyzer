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

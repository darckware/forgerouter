from __future__ import annotations

import os
from typing import Any

import httpx

from app.registry import ProviderModel


def build_chat_payload(
    model: ProviderModel,
    messages: list[dict[str, Any]],
    temperature: float | None = None,
    max_tokens: int | None = None,
    tools: list[dict[str, Any]] | None = None,
    stream: bool = False,
) -> dict[str, Any]:
    payload: dict[str, Any] = {"model": model.provider_model, "messages": messages}
    if temperature is not None:
        payload["temperature"] = temperature
    if max_tokens is not None:
        payload["max_tokens"] = max_tokens
    if tools:
        payload["tools"] = tools
    if stream:
        payload["stream"] = True
        # Ask OpenAI-compatible providers to append a final chunk with token usage,
        # so streamed requests are accounted in route_events like non-streamed ones.
        payload["stream_options"] = {"include_usage": True}
    return payload


def build_embeddings_payload(
    model: ProviderModel,
    input_value: Any,
    encoding_format: str | None = None,
    dimensions: int | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {"model": model.provider_model, "input": input_value}
    if encoding_format is not None:
        payload["encoding_format"] = encoding_format
    if dimensions is not None:
        payload["dimensions"] = dimensions
    return payload


def headers_for_model(model: ProviderModel) -> dict[str, str]:
    headers = {"Content-Type": "application/json"}
    from app.providers.plans import plan_for

    plan = plan_for(model.base_url)
    api_key = ""
    if plan and plan.resolve_token:
        api_key = plan.resolve_token(model.api_key or "", model.api_key_env or "")
    api_key = api_key or model.api_key or (os.environ.get(model.api_key_env) if model.api_key_env else None)
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    # Subscription providers may need provider-specific headers (auth_config.extra_headers).
    headers.update(getattr(model, "extra_headers", None) or {})
    return headers


def parse_retry_after(value: str | None) -> int | None:
    """Seconds from a Retry-After header, capped at 6h. HTTP-date form (rare
    from LLM providers) and non-positive values are ignored."""
    if not value:
        return None
    try:
        seconds = int(float(value.strip()))
    except ValueError:
        return None
    if seconds <= 0:
        return None
    return min(seconds, 6 * 3600)


def _attach_retry_after(body: Any, headers: Any) -> Any:
    # Internal marker consumed (and removed) by the router: turns the provider's
    # Retry-After into the model's runtime cooldown instead of the fixed window.
    retry_after = parse_retry_after(headers.get("retry-after"))
    if retry_after and isinstance(body, dict):
        body["_proxyrouter_retry_after"] = retry_after
    return body


def chat_completion(model: ProviderModel, payload: dict[str, Any], timeout: float = 60.0) -> tuple[int, Any]:
    # Subscription plans with their own protocol (e.g. OpenAI Codex) route through
    # their plan handler instead of the default OpenAI-compatible path.
    from app.providers.plans import plan_for

    plan = plan_for(model.base_url)
    if plan and plan.chat_completion:
        return plan.chat_completion(model, payload, timeout=max(timeout, 120.0))
    # Providers registered with the Anthropic wire format go through the generic
    # Messages API client; everything downstream still sees chat completions.
    if getattr(model, "api_format", "openai") == "anthropic":
        from app.providers.anthropic_compatible import anthropic_chat_completion

        return anthropic_chat_completion(model, payload, timeout=max(timeout, 120.0))
    # Local providers (e.g. Ollama) running on host CPU need a higher timeout ceiling
    # so first-run model loading or CPU inference does not trigger a ReadTimeout.
    if getattr(model, "access_type", "") == "local" or "11434" in model.base_url:
        timeout = max(timeout, 180.0)
    url = model.base_url.rstrip("/") + "/chat/completions"
    headers = headers_for_model(model)
    if payload.get("stream"):
        client = httpx.Client(timeout=timeout)
        try:
            request = client.build_request("POST", url, headers=headers, json=payload)
            response = client.send(request, stream=True)
            if response.status_code >= 400:
                try:
                    response.read()
                    body = response.json()
                except Exception:
                    body = {"error": {"message": response.text, "type": "invalid_json"}}
                response.close()
                client.close()
                return response.status_code, _attach_retry_after(body, response.headers)

            def chunk_generator():
                try:
                    for chunk in response.iter_bytes():
                        yield chunk
                finally:
                    response.close()
                    client.close()

            return response.status_code, chunk_generator()
        except Exception as exc:
            client.close()
            raise exc
    else:
        response = httpx.post(url, headers=headers, json=payload, timeout=timeout)
        try:
            body = response.json()
        except Exception:
            body = {"error": {"message": response.text, "type": "invalid_json"}}
        if response.status_code >= 400:
            body = _attach_retry_after(body, response.headers)
        return response.status_code, body


def embeddings(model: ProviderModel, payload: dict[str, Any], timeout: float = 60.0) -> tuple[int, Any]:
    # Embeddings only exist on the plain OpenAI-compatible /embeddings surface —
    # none of the CLI-auth plan handlers (Codex, Antigravity, Claude Code, z.ai,
    # DeepSeek Web) or the Anthropic Messages API expose one. Report it as an
    # ordinary provider error rather than raising, so the router's normal
    # candidate fallback moves on instead of the request failing outright —
    # a model tagged "embedding" behind one of these should simply never be a
    # candidate in practice, but this keeps that assumption from being load-bearing.
    from app.providers.plans import plan_for

    plan = plan_for(model.base_url)
    if plan or getattr(model, "api_format", "openai") == "anthropic":
        return 501, {"error": {"message": f"{model.provider} does not support embeddings", "type": "unsupported_capability"}}
    url = model.base_url.rstrip("/") + "/embeddings"
    headers = headers_for_model(model)
    response = httpx.post(url, headers=headers, json=payload, timeout=timeout)
    try:
        body = response.json()
    except Exception:
        body = {"error": {"message": response.text, "type": "invalid_json"}}
    if response.status_code >= 400:
        body = _attach_retry_after(body, response.headers)
    return response.status_code, body

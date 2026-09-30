"""LiteLLM gateway with hierarchical, free-first model routing.

system specification Standard #3 / #6:
  frontier reasoning  -> Groq Llama 3.3 70B  -> Gemini Flash -> Ollama local
  SLM classification  -> Groq Llama 3.1 8B   -> Gemini Flash -> Ollama local

Two behaviours matter beyond plain routing:

1. Rate-limit shedding. Groq's free tier is 30 RPM, the hardest constraint in
   the design. A 429 (or a locally-predicted breach of the bucket) demotes the
   provider immediately rather than retrying into the wall.
2. Offline determinism. With no keys and no Ollama, the gateway answers from a
   deterministic stub so the graph, API, and tests still execute end to end.
   Every stubbed result is flagged `stubbed=True` and surfaced in /health --
   it is never presented to a caller as real model output.
"""

from __future__ import annotations

import asyncio
import json
import re
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any, Literal

from app.config import Settings, get_settings
from app.services.tracing import span
from app.utils.logging import get_logger

log = get_logger(__name__)

Tier = Literal["slm", "frontier"]
Purpose = Literal[
    "intent", "sentiment", "answer", "summary", "rewrite_query", "verify_relevance", "generic"
]

Message = dict[str, str]


@dataclass(slots=True)
class LLMResult:
    text: str
    provider: str
    model: str
    latency_ms: float
    stubbed: bool = False
    usage: dict[str, int] = field(default_factory=dict)

    def json(self, default: Any = None) -> Any:
        """Parse the response as JSON, tolerating fences and surrounding prose."""
        return parse_json(self.text, default=default)


def parse_json(text: str, default: Any = None) -> Any:
    """Best-effort JSON extraction from an LLM response."""
    if not text:
        return default
    candidate = text.strip()

    fence = re.search(r"```(?:json)?\s*([\s\S]*?)```", candidate)
    if fence:
        candidate = fence.group(1).strip()

    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        pass

    # Fall back to the outermost {...} or [...] span.
    for opener, closer in (("{", "}"), ("[", "]")):
        start = candidate.find(opener)
        end = candidate.rfind(closer)
        if start != -1 and end > start:
            try:
                return json.loads(candidate[start : end + 1])
            except json.JSONDecodeError:
                continue
    return default


class _RateBucket:
    """Sliding-window request counter guarding a provider's free-tier RPM."""

    __slots__ = ("_hits", "_limit", "_window")

    def __init__(self, limit: int, window_seconds: float = 60.0) -> None:
        self._limit = limit
        self._window = window_seconds
        self._hits: deque[float] = deque()

    def _prune(self, now: float) -> None:
        cutoff = now - self._window
        while self._hits and self._hits[0] < cutoff:
            self._hits.popleft()

    def allow(self) -> bool:
        now = time.monotonic()
        self._prune(now)
        return len(self._hits) < self._limit

    def record(self) -> None:
        self._hits.append(time.monotonic())

    def reset(self) -> None:
        """Clear the window after waiting it out."""
        self._hits.clear()

    def penalise(self) -> None:
        """A real 429 -- saturate the window so we stop trying this provider."""
        now = time.monotonic()
        self._prune(now)
        while len(self._hits) < self._limit:
            self._hits.append(now)


@dataclass(slots=True)
class _Provider:
    name: str
    model: str
    api_key: str | None
    api_base: str | None
    bucket: _RateBucket


class LLMGateway:
    """Async, fallback-chained access to the configured free-tier models."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self._ollama_available: bool | None = None
        self._announced_stub_mode = False
        self._buckets = {
            "groq": _RateBucket(limit=28),  # 30 RPM tier, 2 held back as headroom
            "gemini": _RateBucket(limit=16),  # measured 20 RPM free tier, 4 held back
            "ollama": _RateBucket(limit=10_000),
        }

    # --- provider chain -------------------------------------------------
    def _chain(self, tier: Tier) -> list[_Provider]:
        s = self.settings
        chain: list[_Provider] = []

        if s.has_groq:
            chain.append(
                _Provider(
                    name="groq",
                    model=s.groq_slm_model if tier == "slm" else s.groq_frontier_model,
                    api_key=s.groq_api_key.get_secret_value() if s.groq_api_key else None,
                    api_base=None,
                    bucket=self._buckets["groq"],
                )
            )
        if s.has_gemini:
            chain.append(
                _Provider(
                    name="gemini",
                    model=s.gemini_model,
                    api_key=s.gemini_api_key.get_secret_value() if s.gemini_api_key else None,
                    api_base=None,
                    bucket=self._buckets["gemini"],
                )
            )
        if self._ollama_available is not False:
            chain.append(
                _Provider(
                    name="ollama",
                    model=s.ollama_model,
                    api_key=None,
                    api_base=s.ollama_base_url,
                    bucket=self._buckets["ollama"],
                )
            )
        return chain

    async def probe_ollama(self) -> bool:
        """Cache whether a local Ollama is reachable (called once at startup)."""
        if self._ollama_available is not None:
            return self._ollama_available
        import httpx

        try:
            async with httpx.AsyncClient(timeout=1.5) as client:
                resp = await client.get(f"{self.settings.ollama_base_url}/api/tags")
                self._ollama_available = resp.status_code == 200
        except Exception:
            self._ollama_available = False
        return bool(self._ollama_available)

    # --- public API -----------------------------------------------------
    async def complete(
        self,
        messages: list[Message],
        *,
        tier: Tier = "frontier",
        purpose: Purpose = "generic",
        temperature: float = 0.2,
        max_tokens: int = 512,
        json_mode: bool = False,
        on_delta: Callable[[str], None] | None = None,
    ) -> LLMResult:
        """Run `messages` through the fallback chain; stub if nothing answers.

        `on_delta` opts into token streaming: it is called with each fragment as
        it arrives. Passing it changes only *how* the text is fetched -- the
        provider chain, rate buckets, and stub fallback are unchanged, so a
        streaming call degrades exactly like a buffered one.
        """
        started = time.perf_counter()
        errors: list[str] = []

        for provider in self._chain(tier):
            if not provider.bucket.allow():
                if self.settings.llm_wait_on_rate_limit:
                    # Evaluation mode: pace to the free-tier limit rather than
                    # degrading to the stub and reporting it as a model result.
                    wait = self.settings.llm_rate_limit_wait_seconds
                    log.info(
                        "llm.rate_limit_wait",
                        extra={"provider": provider.name, "seconds": wait},
                    )
                    await asyncio.sleep(wait)
                    provider.bucket.reset()
                else:
                    errors.append(f"{provider.name}: local rate bucket exhausted")
                    log.warning(
                        "llm.provider_shed",
                        extra={"provider": provider.name, "reason": "rate_bucket"},
                    )
                    continue
            try:
                provider.bucket.record()
                with span(
                    f"llm.{provider.name}",
                    kind="generation",
                    provider=provider.name,
                    model=provider.model,
                    tier=tier,
                    purpose=purpose,
                ) as generation:
                    text, usage = await self._call(
                        provider,
                        messages,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        json_mode=json_mode,
                        tier=tier,
                        on_delta=on_delta,
                    )
                    generation.metadata.update(usage)
                if not text.strip():
                    raise RuntimeError("empty completion")
                return LLMResult(
                    text=text,
                    provider=provider.name,
                    model=provider.model,
                    latency_ms=(time.perf_counter() - started) * 1000,
                    usage=usage,
                )
            except Exception as exc:
                message = str(exc)
                errors.append(f"{provider.name}: {message[:160]}")
                if _is_rate_limit(exc):
                    provider.bucket.penalise()
                if provider.name == "ollama":
                    self._ollama_available = False
                log.warning(
                    "llm.provider_failed",
                    extra={
                        "provider": provider.name,
                        "model": provider.model,
                        "error": message[:300],
                    },
                )

        # Fully-offline is a supported configuration, not an incident: announce
        # the degradation once, then stay quiet instead of an ERROR per call.
        if not self._chain(tier):
            if not self._announced_stub_mode:
                self._announced_stub_mode = True
                log.warning("llm.stub_mode_engaged", extra={"reason": "no provider configured"})
        else:
            log.error("llm.all_providers_failed", extra={"errors": errors, "purpose": purpose})

        with span("llm.stub", kind="generation", provider="stub", purpose=purpose):
            return self._stub(
                messages,
                purpose=purpose,
                latency_ms=(time.perf_counter() - started) * 1000,
            )

    async def _call(
        self,
        provider: _Provider,
        messages: list[Message],
        *,
        temperature: float,
        max_tokens: int,
        json_mode: bool,
        tier: Tier = "frontier",
        on_delta: Callable[[str], None] | None = None,
    ) -> tuple[str, dict[str, int]]:
        # Imported lazily: litellm costs ~2s of import time we do not want at boot.
        import litellm

        litellm.suppress_debug_info = True
        litellm.drop_params = True  # silently drop params a provider rejects

        kwargs: dict[str, Any] = {
            "model": provider.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "timeout": self.settings.llm_timeout_seconds,
            "num_retries": 0,  # the chain is our retry policy
        }
        if provider.api_key:
            kwargs["api_key"] = provider.api_key
        if provider.api_base:
            kwargs["api_base"] = provider.api_base
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}

        # Thinking models (Gemini 3.x) spend the max_tokens budget on reasoning
        # before emitting any text, so a 120-token classification call returns
        # empty content and finish_reason="length". Classification and relevance
        # judging do not benefit from reasoning anyway, and system specification wants the
        # SLM tier fast, so it is switched off there. Providers that do not
        # understand the parameter drop it (litellm.drop_params).
        kwargs["reasoning_effort"] = "none" if tier == "slm" else "low"

        if on_delta is not None:
            return await self._call_streaming(litellm, kwargs, on_delta)

        response = await asyncio.wait_for(
            litellm.acompletion(**kwargs),
            timeout=self.settings.llm_timeout_seconds + 2,
        )
        text = response.choices[0].message.content or ""
        raw_usage = getattr(response, "usage", None)
        usage: dict[str, int] = {}
        if raw_usage is not None:
            usage = {
                "prompt_tokens": int(getattr(raw_usage, "prompt_tokens", 0) or 0),
                "completion_tokens": int(getattr(raw_usage, "completion_tokens", 0) or 0),
            }
        return text, usage

    async def _call_streaming(
        self,
        litellm: Any,
        kwargs: dict[str, Any],
        on_delta: Callable[[str], None],
    ) -> tuple[str, dict[str, int]]:
        """Consume a streamed completion, forwarding each fragment as it lands.

        The deadline covers the whole stream, not each chunk: a provider that
        starts fast and then stalls mid-sentence should still be abandoned.
        """
        kwargs = {**kwargs, "stream": True, "stream_options": {"include_usage": True}}
        parts: list[str] = []
        usage: dict[str, int] = {}

        async def consume() -> None:
            stream = await litellm.acompletion(**kwargs)
            async for chunk in stream:
                choices = getattr(chunk, "choices", None) or []
                if choices:
                    fragment = getattr(choices[0].delta, "content", None)
                    if fragment:
                        parts.append(fragment)
                        on_delta(fragment)
                # Usage arrives on the final chunk when include_usage is honoured.
                raw = getattr(chunk, "usage", None)
                if raw is not None:
                    usage["prompt_tokens"] = int(getattr(raw, "prompt_tokens", 0) or 0)
                    usage["completion_tokens"] = int(getattr(raw, "completion_tokens", 0) or 0)

        await asyncio.wait_for(consume(), timeout=self.settings.llm_timeout_seconds + 2)
        return "".join(parts), usage

    # --- offline stub ---------------------------------------------------
    def _stub(self, messages: list[Message], *, purpose: Purpose, latency_ms: float) -> LLMResult:
        """Deterministic offline responses, dispatched by call-site purpose."""
        user_text = next((m["content"] for m in reversed(messages) if m.get("role") == "user"), "")
        handler = _STUB_HANDLERS.get(purpose, _stub_generic)
        return LLMResult(
            text=handler(user_text, messages),
            provider="stub",
            model="offline-stub",
            latency_ms=latency_ms,
            stubbed=True,
        )

    def status(self) -> dict[str, Any]:
        """Provider availability for /health."""
        return {
            "groq": self.settings.has_groq,
            "gemini": self.settings.has_gemini,
            "ollama": self._ollama_available,
            "degraded_to_stub": not (
                self.settings.has_any_hosted_llm or bool(self._ollama_available)
            ),
        }


def _is_rate_limit(exc: Exception) -> bool:
    text = f"{type(exc).__name__} {exc}".lower()
    return "ratelimit" in text or "rate limit" in text or "429" in text


def retry_after_seconds(exc: Exception) -> float | None:
    """Seconds the provider asked us to wait, if it said.

    Gemini returns "Please retry in 18.000263152s" in the 429 body. Honouring
    that beats a blanket sleep: too short and we 429 again, too long and a
    paced evaluation takes far longer than it needs to.
    """
    match = re.search(r"retry in ([0-9]+(?:\.[0-9]+)?)s", str(exc), re.IGNORECASE)
    if match:
        return float(match.group(1))
    match = re.search(r"retry[-_ ]after[\"':= ]+([0-9]+)", str(exc), re.IGNORECASE)
    return float(match.group(1)) if match else None


# --- Stub handlers ------------------------------------------------------
# Keyword heuristics standing in for SLM/frontier calls when fully offline.

_INTENT_KEYWORDS: dict[str, tuple[str, ...]] = {
    "billing": (
        "bill",
        "billing",
        "charge",
        "charged",
        "invoice",
        "payment",
        "refund",
        "credit",
        "overcharge",
        "subscription",
        "price",
        "pricing",
        "receipt",
    ),
    "technical": (
        "error",
        "broken",
        "crash",
        "bug",
        "login",
        "log in",
        "password",
        "reset",
        "not working",
        "cannot",
        "fail",
        "issue",
        "slow",
        "connect",
        "authenticator",
        "2fa",
        "two factor",
        "verification code",
        "rejected",
        "locked",
        "sign in",
        "signin",
        "sync",
        "blank screen",
        "crashes",
        "closes",
        "freeze",
        "frozen",
        "unresponsive",
        "loading",
    ),
    "returns": (
        "return",
        "refund label",
        "exchange",
        "send back",
        "rma",
        "damaged",
        "wrong item",
        "warranty",
        "defect",
        "defective",
        "faulty",
    ),
    "general": (
        "hours",
        "open",
        "contact",
        "privacy",
        "personal data",
        "gdpr",
        "delete my account",
        "close my account",
        "policy",
        "terms",
    ),
    "orders": (
        "order",
        "shipping",
        "delivery",
        "tracking",
        "shipped",
        "package",
        "where is",
    ),
}

_COMPLEX_MARKERS = (
    "dispute",
    "escalate",
    "manager",
    "supervisor",
    "lawyer",
    "legal",
    "fraud",
    "unauthorized",
    "unauthorised",
    "multiple",
    "still not",
    "third time",
    "cancel my account",
    "chargeback",
)

_NEGATIVE_WORDS = (
    "angry",
    "furious",
    "terrible",
    "awful",
    "horrible",
    "useless",
    "worst",
    "hate",
    "frustrated",
    "frustrating",
    "ridiculous",
    "unacceptable",
    "disgusted",
    "scam",
    "disappointed",
    "annoyed",
    "upset",
    "poor",
    "broken",
    "waiting",
)
_POSITIVE_WORDS = (
    "thanks",
    "thank",
    "great",
    "perfect",
    "excellent",
    "appreciate",
    "helpful",
    "wonderful",
    "awesome",
    "happy",
    "pleased",
    "brilliant",
)


@lru_cache(maxsize=512)
def _keyword_pattern(keyword: str) -> re.Pattern[str]:
    """Whole-word matcher for one keyword or phrase.

    Naive substring matching made "rma" match "pe(rma)nently", so
    "cancel my account permanently" classified as a return.
    """
    # Allow common inflections so "refunds" still matches "refund"; without
    # this, word boundaries would reject every plural and past tense.
    return re.compile(r"\b" + re.escape(keyword) + r"(?:s|es|ed|ing)?\b")


def _stub_intent(user_text: str, _messages: list[Message]) -> str:
    lowered = user_text.lower()
    scores = {
        intent: sum(1 for kw in words if _keyword_pattern(kw).search(lowered))
        for intent, words in _INTENT_KEYWORDS.items()
    }
    best = max(scores, key=lambda k: scores[k])
    hits = scores[best]
    intent = best if hits else "general"
    # Calibrated against the 0.70 escalation threshold: one unambiguous keyword
    # clears it, no evidence falls below it. A vague message escalating to a
    # human is correct; a clear FAQ escalating is not.
    confidence = 0.62 + min(hits, 3) * 0.11 if hits else 0.45
    is_complex = any(marker in lowered for marker in _COMPLEX_MARKERS) or (
        len(user_text.split()) > 60
    )
    return json.dumps(
        {
            "intent": intent,
            "confidence": round(min(confidence, 0.94), 2),
            "resolution_path": "complex" if is_complex else "simple",
            "reasoning": "offline keyword heuristic",
        }
    )


def _stub_sentiment(user_text: str, _messages: list[Message]) -> str:
    lowered = user_text.lower()
    neg = sum(1 for w in _NEGATIVE_WORDS if w in lowered)
    pos = sum(1 for w in _POSITIVE_WORDS if w in lowered)
    raw = (pos - neg) / max(pos + neg, 1)
    if "!" in user_text and neg:
        raw -= 0.2
    letters = [c for c in user_text if c.isalpha()]
    if len(letters) > 12 and all(c.isupper() for c in letters):
        raw -= 0.2  # shouting
    score = max(-1.0, min(1.0, raw))
    return json.dumps(
        {
            "sentiment_score": round(score, 2),
            "emotion": "frustrated" if score < -0.3 else "neutral" if score < 0.3 else "positive",
            "reasoning": "offline lexicon heuristic",
        }
    )


def _stub_verify_relevance(_user_text: str, _messages: list[Message]) -> str:
    # Offline we cannot judge semantics; accept and let retrieval scores rank.
    return json.dumps({"relevant": True, "reasoning": "offline stub accepts retrieval ordering"})


def _stub_rewrite_query(user_text: str, _messages: list[Message]) -> str:
    words = [w for w in re.findall(r"[a-z0-9]+", user_text.lower()) if len(w) > 3]
    return " ".join(dict.fromkeys(words))[:180] or user_text


def _stub_answer(user_text: str, messages: list[Message]) -> str:
    from app.agents.prompts.system import KB_SENTINEL

    block = next(
        (
            m["content"]
            for m in messages
            if m.get("role") == "system" and KB_SENTINEL in m["content"]
        ),
        "",
    )
    excerpts = block.split(KB_SENTINEL, 1)[-1].strip()
    if excerpts and excerpts != "none matched this question.":
        # Surface the retrieved knowledge verbatim rather than inventing prose.
        first = excerpts.split("\n\n")[0][:700]
        return (
            "Here is what our knowledge base says about that:\n\n"
            f"{first}\n\n"
            "Note: no language model is connected to this deployment, so this is the "
            "source material rather than a tailored answer. I can bring in a human "
            "colleague if you would like."
        )
    return (
        "Thanks for reaching out. I have recorded your request: "
        f"{user_text[:200]!r}. "
        "No language model is currently connected to this deployment, so I cannot "
        "resolve this automatically. I can hand you to a human colleague with the "
        "full context of this conversation."
    )


def _stub_summary(user_text: str, _messages: list[Message]) -> str:
    return f"Customer contacted support regarding: {user_text[:240]}"


def _stub_generic(user_text: str, _messages: list[Message]) -> str:
    return f"[offline stub] Received: {user_text[:200]}"


_STUB_HANDLERS: dict[str, Any] = {
    "intent": _stub_intent,
    "sentiment": _stub_sentiment,
    "answer": _stub_answer,
    "summary": _stub_summary,
    "rewrite_query": _stub_rewrite_query,
    "verify_relevance": _stub_verify_relevance,
    "generic": _stub_generic,
}


_gateway: LLMGateway | None = None


def get_gateway() -> LLMGateway:
    global _gateway
    if _gateway is None:
        _gateway = LLMGateway()
    return _gateway

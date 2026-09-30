"""Application configuration.

All tunables come from the environment via pydantic-settings. Nothing in the
codebase may hardcode a model name, endpoint, key, or threshold (system specification
Constraints).
"""

from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic import SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=(".env", "../.env"),
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # --- FastAPI ---
    fastapi_env: Literal["development", "staging", "production"] = "development"
    fastapi_secret_key: SecretStr = SecretStr("dev-only-change-me")
    cors_origins: str = "http://localhost:3000"

    # --- LiteLLM gateway ---
    litellm_master_key: SecretStr | None = None
    litellm_log_level: str = "INFO"

    # --- Model routing: Groq (primary) -> Gemini (failover) -> Ollama (local) ---
    groq_api_key: SecretStr | None = None
    groq_frontier_model: str = "groq/llama-3.3-70b-versatile"
    groq_slm_model: str = "groq/llama-3.1-8b-instant"
    groq_stt_model: str = "whisper-large-v3"

    gemini_api_key: SecretStr | None = None
    # Pinned, after the floating alias turned out to fail in the worse of the
    # two available ways.
    #
    # History: gemini-2.0-flash (the system specification default) was retired
    # and returned 404, so this moved to `gemini-flash-latest` on the reasoning
    # that an alias survives retirement. That alias then began accepting
    # generateContent requests and never answering them -- measured, the model
    # list returned in 0.9s while generation hung past 120s. A pinned model that
    # is retired fails in ~1s with a 404 and the provider chain sheds
    # immediately; an alias that stalls burns the full llm_timeout_seconds on
    # every call, which at four calls a turn was 80s of dead time before the
    # whole turn degraded to the offline stub.
    #
    # Fast failure is worth more here than surviving a retirement, so this is
    # pinned. If it 404s, the chain moves on in a second and this line is the
    # thing to change.
    gemini_model: str = "gemini/gemini-2.5-flash"

    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "ollama/qwen2.5:32b"

    # Production sheds when a free-tier bucket is exhausted: a customer must not
    # wait on a rate limit. Evaluation is the opposite -- a shed call silently
    # becomes stub output and corrupts the measurement -- so the harness turns
    # this on to pace itself instead.
    llm_wait_on_rate_limit: bool = False
    llm_rate_limit_wait_seconds: float = 65.0

    llm_timeout_seconds: float = 20.0
    llm_max_retries: int = 2

    # --- Voice ---
    pipecat_log_level: str = "INFO"
    livekit_url: str = "ws://localhost:7880"
    livekit_api_key: SecretStr | None = None
    livekit_api_secret: SecretStr | None = None
    kokoro_model_path: str = "./models/kokoro-82m"
    kokoro_voice: str = "af_heart"
    kokoro_lang: str = "en-us"

    # --- Telephony ---
    twilio_account_sid: str | None = None
    twilio_auth_token: SecretStr | None = None
    twilio_phone_number: str | None = None
    # The public URL Twilio calls. Signature validation hashes the URL Twilio
    # signed, which behind a tunnel or TLS terminator is NOT the URL this app
    # sees -- so it is configuration, not inference.
    twilio_webhook_base_url: str | None = None

    # --- Email channel ---
    # Provider webhook signing key (Mailgun-style HMAC-SHA256). Without it,
    # inbound email is refused: a forged From: header is trivial, so the
    # provider signature is the only trustworthy signal.
    email_webhook_secret: SecretStr | None = None
    email_domain: str | None = None
    email_from_address: str | None = None

    # --- Qdrant ---
    qdrant_host: str = "localhost"
    qdrant_grpc_port: int = 6334
    qdrant_rest_port: int = 6333
    qdrant_collection: str = "support-knowledge"
    qdrant_prefer_grpc: bool = True

    # --- Embeddings ---
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    embedding_dim: int = 384

    # --- Persistence ---
    database_url: str | None = None
    redis_url: str | None = None
    # Budget for a single database call. A store that has not answered in this
    # long is treated as unavailable, so the in-memory fallback engages while
    # someone is still watching -- rather than after the OS gives up on the
    # socket, which measured at 260 seconds.
    db_timeout_seconds: float = 5.0

    # --- CRM (via MCP) ---
    crm_provider: str | None = None
    crm_api_key: SecretStr | None = None
    crm_base_url: str | None = None

    # --- Delivery advisories (Tavily) ---
    # Optional: when absent the Order Agent reports only verified carrier data
    # and never implies that a web disruption search was performed.
    tavily_api_key: SecretStr | None = None

    # --- Observability ---
    langfuse_public_key: SecretStr | None = None
    langfuse_secret_key: SecretStr | None = None
    langfuse_host: str = "http://localhost:3001"

    # Instrumentation runs whenever enabled; the backend is Langfuse if
    # configured, otherwise an in-memory recorder inspectable at /api/traces.
    tracing_enabled: bool = True
    # Spans carry metadata only by default. Traces outlive conversations, and
    # raw message text must not be persisted (system specification Constraints).
    tracing_capture_content: bool = False

    # --- Security / policy ---
    pii_masking_enabled: bool = True
    escalation_refund_threshold: float = 100.00

    # When true, a client-supplied customer_id that matches a known account is
    # accepted as proof of identity. Convenient for local development; MUST be
    # false in production, where identity has to arrive as a verified JWT claim
    # or a carrier-verified caller ID from middleware.
    auth_trust_channel_identity: bool = True

    # --- SLA: first-response targets for escalated conversations (minutes) ---
    # Configurable because every support contract differs; hardcoding these
    # would be a guess presented as policy.
    sla_urgent_minutes: int = 15
    sla_high_minutes: int = 60
    sla_normal_minutes: int = 240
    sla_low_minutes: int = 480

    # --- Handoff notification (SMTP) ---
    # When a conversation escalates, the human queue is told. Without a host
    # configured, the intent is logged and reported as not sent -- never
    # claimed as delivered.
    smtp_host: str | None = None
    smtp_port: int = 587
    smtp_username: str | None = None
    smtp_password: SecretStr | None = None
    smtp_use_tls: bool = True
    smtp_from_address: str | None = None
    # Where handoff alerts go: the support queue address, comma separated.
    handoff_notify_emails: str = ""

    # --- Demo data ---
    # The seeded CRM records are fixtures for local development. Turn this off
    # and the CRM starts empty, so nothing fabricated can be mistaken for a
    # real customer.
    seed_demo_data: bool = True

    # --- Escalation tuning (system specification rule engine) ---
    sentiment_window_size: int = 5
    sentiment_escalation_threshold: float = -0.3
    confidence_escalation_threshold: float = 0.70
    max_tool_attempts: int = 2

    # --- Voice SLO ---
    target_ttfa_ms: int = 400

    # --- MCP ---
    mcp_tool_timeout_seconds: float = 3.0

    # --- Rate limiting ---
    rate_limit_requests: int = 60
    rate_limit_window_seconds: int = 60

    @field_validator("cors_origins")
    @classmethod
    def _strip_origins(cls, v: str) -> str:
        return v.strip()

    @model_validator(mode="after")
    def _harden_production(self) -> Settings:
        """Refuse to trust client-asserted identity in production."""
        if self.fastapi_env == "production" and self.auth_trust_channel_identity:
            self.auth_trust_channel_identity = False
        return self

    @property
    def cors_origin_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]

    # --- Capability probes: drive graceful degradation ---
    @property
    def has_groq(self) -> bool:
        return self.groq_api_key is not None and bool(self.groq_api_key.get_secret_value())

    @property
    def has_gemini(self) -> bool:
        return self.gemini_api_key is not None and bool(self.gemini_api_key.get_secret_value())

    @property
    def has_any_hosted_llm(self) -> bool:
        return self.has_groq or self.has_gemini

    @property
    def twilio_webhooks_ready(self) -> bool:
        """Inbound webhooks require both a token to verify with and the signed URL."""
        return bool(
            self.twilio_auth_token
            and self.twilio_auth_token.get_secret_value()
            and self.twilio_webhook_base_url
        )

    @property
    def email_webhooks_ready(self) -> bool:
        """Inbound email needs a signing key and a domain for Message-IDs."""
        return bool(
            self.email_webhook_secret
            and self.email_webhook_secret.get_secret_value()
            and self.email_domain
        )

    @property
    def handoff_notify_list(self) -> list[str]:
        return [e.strip() for e in self.handoff_notify_emails.split(",") if e.strip()]

    @property
    def smtp_ready(self) -> bool:
        """Outbound mail needs a host, a sender, and at least one recipient."""
        return bool(self.smtp_host and self.smtp_from_address and self.handoff_notify_list)

    @property
    def has_postgres(self) -> bool:
        return bool(self.database_url)

    @property
    def has_redis(self) -> bool:
        return bool(self.redis_url)

    @property
    def is_production(self) -> bool:
        return self.fastapi_env == "production"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Cached settings singleton. Call `get_settings.cache_clear()` in tests."""
    return Settings()

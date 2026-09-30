"""Alembic migrations and the ORM schema.

These run the real migration against a temporary SQLite database, so the
migration is executed rather than merely present. Portable column types are what
make that possible without Docker -- see docs/DEVIATIONS.md for the JSONB
trade-off that buys.
"""

from __future__ import annotations

import subprocess
import sys
from collections.abc import Iterator
from pathlib import Path

import pytest
from sqlalchemy import create_engine, inspect

from app.models.tables import (
    Base,
    Conversation,
    Customer,
    FeedbackResponse,
    Message,
    Ticket,
)

pytestmark = pytest.mark.timeout(180)

BACKEND_ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def sqlite_url(tmp_path: Path) -> Iterator[str]:
    db = tmp_path / "migration_test.db"
    yield f"sqlite:///{db.as_posix()}"


def run_alembic(*args: str, url: str) -> subprocess.CompletedProcess[str]:
    """Invoke the alembic CLI the way an operator would."""
    return subprocess.run(
        [sys.executable, "-m", "alembic", *args],
        cwd=BACKEND_ROOT,
        env={
            **_clean_env(),
            "DATABASE_URL": url,
            "PYTHONPATH": str(BACKEND_ROOT),
        },
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )


def _clean_env() -> dict[str, str]:
    import os

    env = dict(os.environ)
    env.pop("DATABASE_URL", None)
    return env


# --- Migration execution ------------------------------------------------
def test_upgrade_head_creates_every_table(sqlite_url: str) -> None:
    result = run_alembic("upgrade", "head", url=sqlite_url)
    assert result.returncode == 0, result.stderr

    inspector = inspect(create_engine(sqlite_url))
    tables = set(inspector.get_table_names())
    assert {
        "customers",
        "conversations",
        "messages",
        "tickets",
        "feedback_responses",
        "alembic_version",
    } <= tables


def test_downgrade_removes_them_again(sqlite_url: str) -> None:
    """A migration that cannot be rolled back is a one-way door."""
    assert run_alembic("upgrade", "head", url=sqlite_url).returncode == 0

    result = run_alembic("downgrade", "base", url=sqlite_url)
    assert result.returncode == 0, result.stderr

    tables = set(inspect(create_engine(sqlite_url)).get_table_names())
    assert "tickets" not in tables
    assert "messages" not in tables


def test_upgrade_is_idempotent(sqlite_url: str) -> None:
    assert run_alembic("upgrade", "head", url=sqlite_url).returncode == 0
    second = run_alembic("upgrade", "head", url=sqlite_url)
    assert second.returncode == 0, second.stderr


def test_schema_matches_the_models_with_no_drift(sqlite_url: str) -> None:
    """Autogenerate must find nothing after upgrading: models and DDL agree."""
    assert run_alembic("upgrade", "head", url=sqlite_url).returncode == 0

    result = run_alembic("check", url=sqlite_url)
    assert result.returncode == 0, (
        "models have drifted from the migrations; run "
        f"`alembic revision --autogenerate`\n{result.stdout}\n{result.stderr}"
    )


def test_migration_fails_loudly_without_a_database_url() -> None:
    """A silent default would migrate the wrong database."""
    result = run_alembic("upgrade", "head", url="")
    assert result.returncode != 0
    assert "DATABASE_URL" in (result.stderr + result.stdout)


# --- Schema shape -------------------------------------------------------
def test_handoff_queue_index_exists(sqlite_url: str) -> None:
    """The queue filters status and orders by recency; without this it scans."""
    assert run_alembic("upgrade", "head", url=sqlite_url).returncode == 0

    indexes = inspect(create_engine(sqlite_url)).get_indexes("tickets")
    names = {i["name"] for i in indexes}
    assert "ix_tickets_status_updated" in names


def test_message_lookup_index_exists(sqlite_url: str) -> None:
    assert run_alembic("upgrade", "head", url=sqlite_url).returncode == 0
    indexes = inspect(create_engine(sqlite_url)).get_indexes("messages")
    assert "ix_messages_conversation_created" in {i["name"] for i in indexes}


def test_metadata_declares_the_expected_tables() -> None:
    assert set(Base.metadata.tables) == {
        "customers",
        "conversations",
        "messages",
        "tickets",
        "feedback_responses",
    }


def test_checkpoints_are_not_modelled_here() -> None:
    """AsyncPostgresSaver owns its own schema; duplicating it guarantees drift."""
    assert not any("checkpoint" in name for name in Base.metadata.tables)


def test_column_types_stay_portable() -> None:
    """JSONB would pin these migrations to PostgreSQL and break SQLite CI."""
    for table in Base.metadata.tables.values():
        for column in table.columns:
            assert "JSONB" not in type(column.type).__name__.upper(), (
                f"{table.name}.{column.name} uses a PostgreSQL-only type"
            )


# --- ORM round trip -----------------------------------------------------
def test_orm_round_trip(sqlite_url: str) -> None:
    from sqlalchemy.orm import Session

    assert run_alembic("upgrade", "head", url=sqlite_url).returncode == 0
    engine = create_engine(sqlite_url)

    with Session(engine) as session:
        session.add(Customer(customer_id="cust_x", name="Test", email="t@x.com", tier="enterprise"))
        session.add(
            Conversation(conversation_id="conv_x", customer_id="cust_x", channels=["webchat"])
        )
        session.add(
            Ticket(
                ticket_id="TKT-1",
                customer_id="cust_x",
                conversation_id="conv_x",
                subject="Refund",
                status="pending_human",
                escalated=True,
                escalation_reason="explicit_keyword:human",
                citations=[{"id": "kb-refund-timing", "relevance": 0.9}],
                handoff_packet={"churn_risk": "high"},
            )
        )
        session.add(
            Message(conversation_id="conv_x", role="customer", text="card [[CC_1]] charged twice")
        )
        session.add(FeedbackResponse(ticket_id="TKT-1", score=5, comment="good"))
        session.commit()

    with Session(engine) as session:
        ticket = session.get(Ticket, "TKT-1")
        assert ticket is not None
        # JSON columns must survive the round trip as structured data.
        assert ticket.citations[0]["id"] == "kb-refund-timing"
        assert ticket.handoff_packet == {"churn_risk": "high"}
        assert ticket.customer.tier == "enterprise"

        conversation = session.get(Conversation, "conv_x")
        assert conversation is not None
        assert conversation.channels == ["webchat"]
        assert "[[CC_1]]" in conversation.messages[0].text
        assert "4111" not in conversation.messages[0].text


def test_cascade_deletes_messages_with_their_conversation(sqlite_url: str) -> None:
    from sqlalchemy.orm import Session

    assert run_alembic("upgrade", "head", url=sqlite_url).returncode == 0
    engine = create_engine(sqlite_url)

    with Session(engine) as session:
        conversation = Conversation(conversation_id="conv_del", customer_id="c")
        conversation.messages.append(
            Message(conversation_id="conv_del", role="customer", text="hi")
        )
        session.add(conversation)
        session.commit()

        session.delete(conversation)
        session.commit()
        assert session.query(Message).filter_by(conversation_id="conv_del").count() == 0

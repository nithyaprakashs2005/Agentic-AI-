"""Portable SQLAlchemy models matching the Alembic-managed schema."""

from __future__ import annotations

from datetime import UTC, datetime

from sqlalchemy import (
    JSON,
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


def _now() -> datetime:
    return datetime.now(UTC)


class Base(DeclarativeBase):
    pass


class Customer(Base):
    __tablename__ = "customers"

    customer_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    name: Mapped[str] = mapped_column(String(200), default="")
    email: Mapped[str] = mapped_column(String(320), default="", index=True)
    phone: Mapped[str] = mapped_column(String(32), default="", index=True)
    tier: Mapped[str] = mapped_column(String(16), default="standard")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)

    tickets: Mapped[list[Ticket]] = relationship(back_populates="customer")


class Conversation(Base):
    __tablename__ = "conversations"

    conversation_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    customer_id: Mapped[str] = mapped_column(String(64), index=True)
    channels: Mapped[list[str]] = mapped_column(JSON, default=list)
    turn_count: Mapped[int] = mapped_column(Integer, default=0)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_now, onupdate=_now
    )

    messages: Mapped[list[Message]] = relationship(
        back_populates="conversation", cascade="all, delete-orphan"
    )


class Message(Base):
    __tablename__ = "messages"
    __table_args__ = (
        Index("ix_messages_conversation_created", "conversation_id", "created_at"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    conversation_id: Mapped[str] = mapped_column(
        ForeignKey("conversations.conversation_id", ondelete="CASCADE"), index=True
    )
    role: Mapped[str] = mapped_column(String(16))
    text: Mapped[str] = mapped_column(Text)
    channel: Mapped[str] = mapped_column(String(16), default="webchat")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)

    conversation: Mapped[Conversation] = relationship(back_populates="messages")


class Ticket(Base):
    __tablename__ = "tickets"
    __table_args__ = (Index("ix_tickets_status_updated", "status", "updated_at"),)

    ticket_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    customer_id: Mapped[str] = mapped_column(ForeignKey("customers.customer_id"), index=True)
    conversation_id: Mapped[str | None] = mapped_column(String(64), index=True, default=None)
    channel: Mapped[str] = mapped_column(String(16), default="webchat")
    subject: Mapped[str] = mapped_column(String(200), default="")
    summary: Mapped[str] = mapped_column(Text, default="")
    status: Mapped[str] = mapped_column(String(24), default="open", index=True)
    priority: Mapped[str] = mapped_column(String(16), default="normal")
    intent: Mapped[str] = mapped_column(String(32), default="general", index=True)
    sentiment_score: Mapped[float] = mapped_column(Float, default=0.0)
    escalated: Mapped[bool] = mapped_column(Boolean, default=False, index=True)
    escalation_reason: Mapped[str | None] = mapped_column(String(64), default=None)
    citations: Mapped[list[dict[str, object]]] = mapped_column(JSON, default=list)
    handoff_packet: Mapped[dict[str, object] | None] = mapped_column(JSON, default=None)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_now, onupdate=_now
    )
    escalated_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), default=None, index=True
    )
    first_responded_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), default=None
    )
    assigned_to: Mapped[str | None] = mapped_column(String(120), default=None)
    channels: Mapped[list[str]] = mapped_column(JSON, default=list)
    notified_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), default=None)

    customer: Mapped[Customer] = relationship(back_populates="tickets")


class FeedbackResponse(Base):
    __tablename__ = "feedback_responses"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    ticket_id: Mapped[str] = mapped_column(ForeignKey("tickets.ticket_id"), index=True)
    conversation_id: Mapped[str | None] = mapped_column(String(64), default=None)
    score: Mapped[int] = mapped_column(Integer)
    comment: Mapped[str] = mapped_column(Text, default="")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now, index=True)

    ticket: Mapped[Ticket] = relationship()
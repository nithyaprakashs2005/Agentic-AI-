"""CRM adapter: customer identity, orders, invoices, tickets.

No live CRM is configured in this environment (`CRM_BASE_URL` unset), so this
module serves a seeded in-memory dataset. It is the single source of truth
behind both the MCP tool implementations and the authentication node, so
swapping in a real provider means implementing `CrmBackend` once.

Identity matching deliberately does NOT trust an arbitrary caller-supplied
`customer_id` in production -- see `Settings.auth_trust_channel_identity`.
"""

from __future__ import annotations

import re
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol

from app.config import Settings, get_settings
from app.utils.logging import get_logger

log = get_logger(__name__)

Tier = Literal["free", "standard", "enterprise"]

_EMAIL_RE = re.compile(r"\b[\w.%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
_PHONE_RE = re.compile(
    r"(?<!\w)(?:\+\d{1,3}[ .-]?)?(?:\(\d{3}\)|\d{3})[ .-]?\d{3}[ .-]?\d{4}(?!\w)"
)


def _digits(value: str) -> str:
    return re.sub(r"\D", "", value)


@dataclass(slots=True)
class CustomerProfile:
    customer_id: str
    name: str
    email: str
    phone: str
    tier: Tier = "standard"
    matched_on: str = "unknown"


@dataclass(slots=True)
class Order:
    order_id: str
    customer_id: str
    status: str
    placed_at: str
    total: float
    currency: str
    items: list[str]
    carrier: str = ""
    tracking: str = ""
    eta: str = ""
    location: str = ""
    delay_reason: str | None = None
    eligible_for_return: bool = True


@dataclass(slots=True)
class Invoice:
    invoice_id: str
    customer_id: str
    amount: float
    currency: str
    issued_at: str
    status: str
    description: str


@dataclass(slots=True)
class Ticket:
    ticket_id: str
    customer_id: str
    channel: str
    subject: str
    conversation_id: str | None = None
    summary: str = ""
    status: Literal["open", "pending_human", "resolved"] = "open"
    priority: Literal["low", "normal", "high", "urgent"] = "normal"
    intent: str = "general"
    sentiment_score: float = 0.0
    escalated: bool = False
    escalation_reason: str | None = None
    # When the customer began waiting on a human. The SLA clock starts here.
    escalated_at: float | None = None
    first_responded_at: float | None = None
    assigned_to: str | None = None
    # Every channel this conversation has touched, in order first seen.
    channels: list[str] = field(default_factory=list)
    notified_at: float | None = None
    citations: list[dict[str, Any]] = field(default_factory=list)
    transcript: list[dict[str, str]] = field(default_factory=list)
    handoff_packet: dict[str, Any] | None = None
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)

    def public(self) -> dict[str, Any]:
        return {
            "ticket_id": self.ticket_id,
            "customer_id": self.customer_id,
            "channel": self.channel,
            "subject": self.subject,
            "summary": self.summary,
            "status": self.status,
            "priority": self.priority,
            "intent": self.intent,
            "sentiment_score": self.sentiment_score,
            "escalated": self.escalated,
            "escalation_reason": self.escalation_reason,
            "escalated_at": self.escalated_at,
            "first_responded_at": self.first_responded_at,
            "assigned_to": self.assigned_to,
            "channels": self.channels or ([self.channel] if self.channel else []),
            "notified_at": self.notified_at,
            "citations": self.citations,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }


class CrmBackend(Protocol):
    async def identify(
        self, *, customer_id: str, utterance: str, channel: str
    ) -> CustomerProfile | None: ...
    async def get_customer(self, customer_id: str) -> CustomerProfile | None: ...
    async def get_orders(self, customer_id: str) -> list[Order]: ...
    async def get_order(self, order_id: str) -> Order | None: ...
    async def get_invoices(self, customer_id: str) -> list[Invoice]: ...
    async def initiate_return(
        self, order_id: str, item: str, reason: str, customer_id: str | None = None
    ) -> dict[str, Any]: ...


# --- Seeded dataset -----------------------------------------------------
def _seed_customers() -> dict[str, CustomerProfile]:
    rows = [
        CustomerProfile(
            "cust_demo", "Demo Customer", "demo@example.com", "+1 555-000-1000", "standard"
        ),
        CustomerProfile(
            "cust_abc123", "Ada Byron", "ada@example.com", "+1 555-010-2020", "enterprise"
        ),
        CustomerProfile(
            "cust_def456", "Raj Patel", "raj@example.com", "+1 555-030-4040", "standard"
        ),
        CustomerProfile("cust_ghi789", "Mei Chen", "mei@example.com", "+1 555-050-6060", "free"),
    ]
    return {c.customer_id: c for c in rows}


def _seed_orders() -> list[Order]:
    orders = [
        Order(
            "ORD-84021",
            "cust_demo",
            "Shipped",
            "2026-08-18",
            129.99,
            "USD",
            ["Wireless keyboard"],
            "UPS",
            "1Z999AA10123456784",
            "2026-08-25",
        ),
        Order(
            "ORD-84105",
            "cust_demo",
            "Delivered",
            "2026-08-09",
            24.50,
            "USD",
            ["USB-C cable, 2m"],
            "USPS",
            "9400111899223197428490",
            "2026-08-14",
        ),
        Order(
            "ORD-90112",
            "cust_abc123",
            "Processing",
            "2026-08-22",
            1847.00,
            "USD",
            ["Standing desk", "Monitor arm"],
            "",
            "",
            "2026-08-29",
        ),
        Order(
            "ORD-77410",
            "cust_def456",
            "Delivered",
            "2026-07-30",
            89.00,
            "USD",
            ["Noise-cancelling earbuds"],
            "FedEx",
            "770123456789",
            "2026-08-04",
            location="Delivered to mailroom",
            eligible_for_return=False,  # > 30 days
        ),
        Order(
            "ORD-10245",
            "cust_demo",
            "Delivered",
            "2026-09-23",
            110.00,
            "USD",
            ["Running Shoes - Size 10"],
            "FedEx",
            "794912345678",
            "2026-09-26",
            location="Delivered to front porch",
            eligible_for_return=True,
        ),
        Order(
            "ORD-99321",
            "cust_demo",
            "In Transit - Delayed",
            "2026-09-20",
            85.50,
            "USD",
            ["Ergonomic Pillow"],
            "FedEx",
            "794812349999",
            "2026-09-25",
            location="Memphis Distribution Hub",
            delay_reason="Severe weather disruption in Memphis area affecting carrier flights",
            eligible_for_return=False,
        ),
    ]

    # A realistic local dataset makes list, ownership, and ambiguity paths
    # testable without reaching a production store. Named demo orders above
    # remain stable for walkthroughs; the additional rows are deterministic.
    catalog = [
        "Cotton duvet cover",
        "Ceramic coffee mug",
        "Laptop stand",
        "Insulated water bottle",
        "Travel backpack",
    ]
    customers = ["cust_demo", "cust_abc123", "cust_def456", "cust_ghi789"]
    statuses = ["Delivered", "Shipped", "Processing", "Out for delivery"]
    for index in range(58):
        status = statuses[index % len(statuses)]
        delivered = status == "Delivered"
        customer_id = customers[index % len(customers)]
        orders.append(
            Order(
                order_id=f"ORD-{70000 + index}",
                customer_id=customer_id,
                status=status,
                placed_at=f"2026-{'08' if index % 2 else '09'}-{(index % 27) + 1:02d}",
                total=round(18.50 + index * 3.75, 2),
                currency="USD",
                items=[catalog[index % len(catalog)]],
                carrier="UPS" if status != "Processing" else "",
                tracking=f"1ZDEMO{70000 + index}" if status != "Processing" else "",
                eta=f"2026-10-{(index % 20) + 1:02d}",
                location="Regional distribution center" if not delivered else "Delivered",
                eligible_for_return=delivered and index % 3 != 0,
            )
        )
    return orders


def _seed_invoices() -> list[Invoice]:
    return [
        Invoice(
            "INV-2026-0812", "cust_demo", 129.99, "USD", "2026-08-18", "paid", "Order ORD-84021"
        ),
        Invoice(
            "INV-2026-0790", "cust_demo", 24.50, "USD", "2026-08-09", "paid", "Order ORD-84105"
        ),
        Invoice(
            "INV-2026-0901",
            "cust_abc123",
            247.50,
            "USD",
            "2026-08-15",
            "paid",
            "Subscription upgrade - Enterprise tier, prorated",
        ),
        Invoice(
            "INV-2026-0902",
            "cust_abc123",
            247.50,
            "USD",
            "2026-08-16",
            "paid",
            "Subscription upgrade - Enterprise tier, prorated",
        ),
    ]


class InMemoryCrm:
    """Seeded CRM. Ticket writes persist for the process lifetime."""

    backend = "in-memory"

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        # Fixtures only when explicitly enabled. With SEED_DEMO_DATA=false the
        # CRM starts empty and every customer comes from a real interaction, so
        # no invented record can be read as a live account.
        seeded = settings.seed_demo_data
        self._seeded = seeded
        self._customers = _seed_customers() if seeded else {}
        self._orders = _seed_orders() if seeded else []
        self._invoices = _seed_invoices() if seeded else []
        self._tickets: dict[str, Ticket] = {}
        self._by_email = {c.email.lower(): c for c in self._customers.values()}
        self._by_phone = {_digits(c.phone): c for c in self._customers.values()}

    # --- identity ---
    async def identify(
        self, *, customer_id: str, utterance: str, channel: str
    ) -> CustomerProfile | None:
        # 1. Channel-asserted identity. Only trusted when configured -- in
        #    production a JWT/caller-ID assertion must come from middleware,
        #    never from a client-supplied body field.
        if customer_id and self._settings.auth_trust_channel_identity:
            known = self._customers.get(customer_id)
            if known:
                return _with_match(known, f"channel_identity:{channel}")

        # 2. Explicit challenge response: an email or phone in the utterance.
        if utterance:
            email = _EMAIL_RE.search(utterance)
            if email:
                found = self._by_email.get(email.group(0).lower())
                if found:
                    return _with_match(found, "email_challenge")
            phone = _PHONE_RE.search(utterance)
            if phone:
                found = self._by_phone.get(_digits(phone.group(0)))
                if found:
                    return _with_match(found, "phone_challenge")
        return None

    async def get_customer(self, customer_id: str) -> CustomerProfile | None:
        return self._customers.get(customer_id)

    async def get_orders(self, customer_id: str) -> list[Order]:
        return [o for o in self._orders if o.customer_id == customer_id]

    async def get_order(self, order_id: str) -> Order | None:
        raw = order_id.strip()
        candidates = {
            raw.upper(),
            raw.upper().lstrip("#"),
            f"#{raw.upper().lstrip('#')}",
            f"ORD-{raw.upper().lstrip('#')}",
        }
        for o in self._orders:
            if o.order_id.upper() in candidates or o.order_id.upper().lstrip("#") in candidates:
                return o
        return None

    async def get_invoices(self, customer_id: str) -> list[Invoice]:
        return [i for i in self._invoices if i.customer_id == customer_id]

    async def initiate_return(
        self,
        order_id: str,
        item: str,
        reason: str,
        customer_id: str | None = None,
    ) -> dict[str, Any]:
        order = await self.get_order(order_id)
        if not order:
            return {"success": False, "error": f"Order {order_id} not found."}
        if customer_id and order.customer_id != customer_id:
            return {
                "success": False,
                "error": f"Order {order_id} does not belong to the verified customer.",
            }
        if order.status.lower() != "delivered":
            return {
                "success": False,
                "error": f"Order {order_id} has status '{order.status}'. Only delivered items can be returned.",
            }
        if not getattr(order, "eligible_for_return", True):
            return {
                "success": False,
                "error": f"Order {order_id} is outside the 30-day return eligibility window.",
            }

        return_id = f"RET-{time.strftime('%Y%m%d')}-{uuid.uuid4().hex[:4].upper()}"
        item_name = item or (order.items[0] if order.items else "Purchased Item")
        return {
            "success": True,
            "return_id": return_id,
            "order_id": order.order_id,
            "item": item_name,
            "refund_amount": order.total,
            "currency": order.currency,
            "reason": reason,
            "label_url": f"https://returns.example.com/labels/{return_id}.pdf",
            "instructions": (
                f"Prepaid shipping label {return_id} generated. Please box the items in original packaging, "
                "attach the label, and hand off to any authorized carrier. Refund will be released upon warehouse scan."
            ),
        }

    # --- tickets ---
    async def create_ticket(
        self,
        *,
        customer_id: str,
        channel: str,
        subject: str,
        intent: str = "general",
        priority: Literal["low", "normal", "high", "urgent"] = "normal",
        conversation_id: str | None = None,
    ) -> Ticket:
        ticket_id = f"TKT-{time.strftime('%Y%m%d')}-{uuid.uuid4().hex[:4].upper()}"
        ticket = Ticket(
            ticket_id=ticket_id,
            customer_id=customer_id,
            channel=channel,
            subject=subject[:160],
            intent=intent,
            priority=priority,
            conversation_id=conversation_id,
        )
        self._tickets[ticket_id] = ticket
        await self._persist(ticket)
        log.info("crm.ticket_created", extra={"ticket_id": ticket_id, "intent": intent})
        return ticket

    async def update_ticket(self, ticket_id: str, **fields: Any) -> Ticket | None:
        ticket = self._tickets.get(ticket_id)
        if ticket is None:
            return None
        for key, value in fields.items():
            if hasattr(ticket, key):
                setattr(ticket, key, value)
        ticket.updated_at = time.time()
        await self._persist(ticket)
        return ticket

    async def _persist(self, ticket: Ticket) -> None:
        """Mirror a ticket to durable storage, if any is configured.

        Never raises into the turn: losing durability on one ticket must not
        fail the customer's request.
        """
        from app.services.repository import get_repository

        repo = get_repository()
        if repo is None:
            return
        try:
            await repo.save_ticket(ticket)
        except Exception:
            log.warning("crm.persist_failed", extra={"ticket_id": ticket.ticket_id})

    async def get_ticket(self, ticket_id: str) -> Ticket | None:
        cached = self._tickets.get(ticket_id)
        if cached is not None:
            return cached

        # Not in this process: it may predate a restart.
        from app.services.repository import get_repository

        repo = get_repository()
        if repo is None:
            return None
        try:
            row = await repo.get_ticket(ticket_id)
        except Exception:
            # Unreachable store. A ticket we do not hold in memory is reported
            # as absent rather than failing the turn -- the caller's 404 is a
            # truthful "not found here", and the conversation still completes.
            log.warning("crm.lookup_from_db_failed", extra={"ticket_id": ticket_id})
            return None
        if row is None:
            return None
        restored = _ticket_from_row(row)
        self._tickets[ticket_id] = restored
        return restored

    async def list_tickets(self, *, limit: int = 50) -> list[Ticket]:
        from app.services.repository import get_repository

        repo = get_repository()
        if repo is not None:
            try:
                rows = await repo.list_tickets(limit=limit)
                # Durable store is authoritative: it includes tickets from
                # before this process started.
                for row in rows:
                    self._tickets.setdefault(row["ticket_id"], _ticket_from_row(row))
                return [_ticket_from_row(row) for row in rows]
            except Exception:
                log.warning("crm.list_from_db_failed")
        return sorted(self._tickets.values(), key=lambda t: -t.updated_at)[:limit]

    def all_customers(self) -> list[CustomerProfile]:
        """Seeded profiles, mirrored into the database so ticket FKs resolve."""
        return list(self._customers.values())

    def describe(self) -> dict[str, Any]:
        from app.services.repository import get_repository

        repo = get_repository()
        return {
            "backend": "postgres" if repo is not None else self.backend,
            "tickets_durable": repo is not None,
            "demo_data_seeded": self._seeded,
            "customers": len(self._customers),
            "orders": len(self._orders),
            "invoices": len(self._invoices),
            "tickets": len(self._tickets),
        }


def _ticket_from_row(row: dict[str, Any]) -> Ticket:
    """Rehydrate a Ticket from a persisted row."""
    return Ticket(
        ticket_id=row["ticket_id"],
        customer_id=row["customer_id"],
        channel=row["channel"],
        escalation_reason=row.get("escalation_reason"),
        escalated_at=row.get("escalated_at"),
        first_responded_at=row.get("first_responded_at"),
        assigned_to=row.get("assigned_to"),
        channels=row.get("channels") or [],
        notified_at=row.get("notified_at"),
        conversation_id=row["conversation_id"],
        subject=row["subject"],
        transcript=row.get("transcript") or [],
        summary=row["summary"],
        status=row["status"],
        priority=row["priority"],
        intent=row["intent"],
        sentiment_score=row["sentiment_score"],
        escalated=row["escalated"],
        citations=row["citations"] or [],
        handoff_packet=row["handoff_packet"],
        created_at=row["created_at"],
        updated_at=row["updated_at"],
    )


def _with_match(profile: CustomerProfile, method: str) -> CustomerProfile:
    return CustomerProfile(
        customer_id=profile.customer_id,
        name=profile.name,
        email=profile.email,
        phone=profile.phone,
        tier=profile.tier,
        matched_on=method,
    )


_crm: InMemoryCrm | None = None


def init_crm(settings: Settings | None = None) -> InMemoryCrm:
    global _crm
    settings = settings or get_settings()
    if settings.crm_base_url:
        # A real provider would be constructed here; the seeded store still
        # backs local development and tests.
        log.warning(
            "crm.remote_not_implemented",
            extra={"provider": settings.crm_provider, "using": "in-memory"},
        )
    _crm = InMemoryCrm(settings)
    log.info("crm.ready", extra=_crm.describe())
    return _crm


def get_crm() -> InMemoryCrm:
    if _crm is None:
        return init_crm()
    return _crm

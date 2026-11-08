from typing import Any

from .config import Settings
from .data_validation import Dataset
from .models import Customer, Draft, Queue, SearchResult, Ticket, ToolCall
from .retrieval import PolicyIndex


class ToolError(Exception):
    pass


class ToolBudgetExceeded(Exception):
    pass


class ToolBox:
    """All tools share one trace and one call budget."""

    def __init__(self, dataset: Dataset, index: PolicyIndex, settings: Settings):
        self.dataset = dataset
        self.index = index
        self.settings = settings
        self.trace: list[ToolCall] = []

    def _start(self) -> None:
        if len(self.trace) >= self.settings.max_tool_calls:
            raise ToolBudgetExceeded(f"tool-call limit of {self.settings.max_tool_calls} reached")

    def _log(self, name: str, arguments: dict[str, Any], ok: bool, summary: str) -> None:
        self.trace.append(ToolCall(name=name, arguments=arguments, succeeded=ok, summary=summary))

    def get_ticket(self, ticket_id: str) -> Ticket:
        self._start()
        ticket = self.dataset.tickets.get(ticket_id)
        if ticket is None:
            self._log("get_ticket", {"ticket_id": ticket_id}, False, "ticket not found")
            raise ToolError(f"ticket '{ticket_id}' was not found")
        self._log("get_ticket", {"ticket_id": ticket_id}, True, ticket.subject)
        return ticket

    def get_customer_context(self, customer_id: str) -> Customer:
        self._start()
        customer = self.dataset.customers.get(customer_id)
        if customer is None:
            self._log("get_customer_context", {"customer_id": customer_id}, False, "not found")
            raise ToolError(f"customer '{customer_id}' was not found")
        summary = f"{customer.name}, {customer.plan}, verified={customer.verified}"
        self._log("get_customer_context", {"customer_id": customer_id}, True, summary)
        return customer

    def search_policy(self, query: str) -> SearchResult:
        self._start()
        result = self.index.search(query, self.settings.top_k, self.settings.min_score)
        sources = [e.source_id for e in result.evidence]
        summary = f"sources={sources}, best_score={result.best_score}"
        self._log("search_policy", {"query": query[:80]}, True, summary)
        return result

    def check_availability(self, queue: str) -> Queue:
        self._start()
        found = self.dataset.queues.get(queue)
        if found is None:
            self._log("check_availability", {"queue": queue}, False, "queue not found")
            raise ToolError(f"queue '{queue}' was not found")
        summary = f"owner={found.owner}, available={found.available}"
        self._log("check_availability", {"queue": queue}, True, summary)
        return found

    def draft_ticket_update(self, ticket_id: str, message: str) -> Draft:
        self._start()
        draft = Draft(kind="ticket_update", ticket_id=ticket_id, message=message)
        self._log("draft_ticket_update", {"ticket_id": ticket_id}, True, "draft created")
        return draft

    def draft_escalation(self, ticket_id: str, queue: str, reason: str) -> Draft:
        self._start()
        target = self.dataset.queues.get(queue)
        if target is None:
            self._log("draft_escalation", {"ticket_id": ticket_id, "queue": queue}, False, "bad queue")
            raise ToolError(f"queue '{queue}' was not found")
        message = (
            f"Escalation for ticket {ticket_id} to queue {queue} "
            f"(owner: {target.owner}, channel: {target.channel}). {reason}"
        )
        draft = Draft(kind="escalation", ticket_id=ticket_id, queue=queue, message=message)
        self._log("draft_escalation", {"ticket_id": ticket_id, "queue": queue}, True, "draft created")
        return draft

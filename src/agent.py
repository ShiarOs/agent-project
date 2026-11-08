from dataclasses import dataclass, field

from .config import Settings
from .data_validation import Dataset
from .memory import MemoryStore
from .models import AgentResult, Customer, Draft, Status, Ticket
from .retrieval import CitationError, PolicyIndex, strip_suspicious, validate_citations
from .tools import ToolBox, ToolBudgetExceeded, ToolError

ESCALATION_HINT = "escalate responsible queue ticket evidence handoff"


class StepBudgetExceeded(Exception):
    pass


@dataclass
class Run:
    toolbox: ToolBox
    steps: int = 0
    used_memory: bool = False
    ticket_id: str | None = None
    best_score: float = 0.0
    evidence_sources: list[str] = field(default_factory=list)
    safety_events: list[str] = field(default_factory=list)


class Agent:
    def __init__(
        self,
        dataset: Dataset,
        settings: Settings,
        memory: MemoryStore | None = None,
        index: PolicyIndex | None = None,
    ):
        self.dataset = dataset
        self.settings = settings
        self.index = index or PolicyIndex(dataset.policies)
        self.memory = memory or MemoryStore()
        if memory is None:
            self.memory.load_seed(dataset.memory)

    def run(
        self, prompt: str, ticket_id: str | None = None, conversation_id: str = "northstar-demo"
    ) -> AgentResult:
        run = Run(toolbox=ToolBox(self.dataset, self.index, self.settings))
        self.memory.add_turn(conversation_id, "user", prompt)
        try:
            result = self._run(run, ticket_id, conversation_id)
        except (StepBudgetExceeded, ToolBudgetExceeded) as error:
            result = self._finish(run, "review_required", f"Stopped safely: {error}. Human review needed.")
        except ToolError as error:
            result = self._finish(run, "review_required", f"A tool failed: {error}. Human review needed.")
        except CitationError as error:
            result = self._finish(run, "review_required", f"Answer rejected: {error}.")
        self.memory.add_turn(conversation_id, "assistant", result.answer)
        return result

    def ask(self, question: str) -> AgentResult:
        """Answer a free policy question without a ticket."""
        run = Run(toolbox=ToolBox(self.dataset, self.index, self.settings))
        try:
            self._step(run)
            clean, removed = strip_suspicious(question)
            if removed:
                run.safety_events.append("suspicious_instruction_removed")
            search = run.toolbox.search_policy(clean)
            run.best_score = search.best_score
            run.evidence_sources = [e.source_id for e in search.evidence]
            if search.removed_suspicious:
                run.safety_events.append("retrieved_instruction_removed")
            if not search.evidence:
                return self._finish(
                    run, "abstained", "The approved policies do not answer this question."
                )
            lines = [f"{e.text} [{e.source_id}]" for e in search.evidence]
            answer = "Approved policy evidence:\n" + "\n".join(lines)
            citations = validate_citations(answer, set(run.evidence_sources))
            return self._finish(run, "answered", answer, citations)
        except (StepBudgetExceeded, ToolBudgetExceeded) as error:
            return self._finish(run, "review_required", f"Stopped safely: {error}.")

    def _step(self, run: Run) -> None:
        run.steps += 1
        if run.steps > self.settings.max_agent_steps:
            raise StepBudgetExceeded(f"step limit of {self.settings.max_agent_steps} reached")

    def _finish(
        self, run: Run, status: Status, answer: str, citations: list[str] | None = None,
        draft: Draft | None = None,
    ) -> AgentResult:
        return AgentResult(
            status=status,
            answer=answer,
            citations=citations or [],
            evidence_sources=run.evidence_sources,
            tool_calls=run.toolbox.trace,
            safety_events=run.safety_events,
            used_memory=run.used_memory,
            ticket_id=run.ticket_id,
            steps=run.steps,
            best_score=run.best_score,
            draft=draft,
        )

    @staticmethod
    def _route(ticket: Ticket, customer: Customer) -> str | None:
        if ticket.category == "incident":
            return "incident-response"
        if ticket.category == "privacy":
            return "privacy"
        if ticket.category == "account" and not customer.verified:
            return "identity-review"
        return None

    def _run(self, run: Run, ticket_id: str | None, conversation_id: str) -> AgentResult:
        tools = run.toolbox

        self._step(run)
        memory = self.memory.recall(conversation_id)
        if ticket_id is None:
            if memory is None or memory.last_ticket_id is None:
                return self._finish(run, "review_required", "No ticket was given and memory has none.")
            ticket_id = memory.last_ticket_id
            run.used_memory = True

        self._step(run)
        ticket = tools.get_ticket(ticket_id)
        run.ticket_id = ticket.ticket_id
        customer = tools.get_customer_context(ticket.customer_id)

        if run.used_memory and memory and memory.last_customer_id != ticket.customer_id:
            run.used_memory = False
            run.safety_events.append("memory_customer_mismatch")
            return self._finish(run, "review_required", "Memory belongs to another customer.")

        self._step(run)
        clean_text, removed = strip_suspicious(f"{ticket.subject}. {ticket.body}")
        if removed:
            run.safety_events.append("suspicious_instruction_removed")
        queue = self._route(ticket, customer)
        query = f"{clean_text} {ESCALATION_HINT}" if queue else clean_text
        search = tools.search_policy(query)
        run.best_score = search.best_score
        run.evidence_sources = [e.source_id for e in search.evidence]
        if search.removed_suspicious:
            run.safety_events.append("retrieved_instruction_removed")

        if not search.evidence:
            return self._finish(
                run, "abstained",
                "The approved policies do not answer this request, so no action was drafted.",
            )

        sources = " ".join(f"[{s}]" for s in run.evidence_sources)
        allowed = set(run.evidence_sources)

        if queue:
            self._step(run)
            owner = tools.check_availability(queue)
            reason = self._escalation_reason(ticket, sources)
            if not owner.available:
                run.safety_events.append("owner_unavailable")
                answer = (
                    f"Queue {queue} has no available owner ({owner.owner}). "
                    f"A human reviewer must pick this up. {sources}"
                )
                return self._finish(run, "review_required", answer, validate_citations(answer, allowed))
            self._step(run)
            draft = tools.draft_escalation(ticket.ticket_id, queue, reason)
            citations = validate_citations(draft.message, allowed)
            self._remember(conversation_id, ticket, "Escalated to " + queue)
            return self._finish(run, "escalated", draft.message, citations, draft)

        self._step(run)
        message = self._update_message(ticket, customer, sources)
        draft = tools.draft_ticket_update(ticket.ticket_id, message)
        citations = validate_citations(draft.message, allowed)
        self._remember(conversation_id, ticket, "Drafted an update for " + ticket.subject)
        return self._finish(run, "answered", draft.message, citations, draft)

    def _remember(self, conversation_id: str, ticket: Ticket, note: str) -> None:
        self.memory.remember(conversation_id, ticket.ticket_id, ticket.customer_id, note)

    @staticmethod
    def _escalation_reason(ticket: Ticket, sources: str) -> str:
        return (
            f"Subject: {ticket.subject}. Customer impact and details: {ticket.body} "
            f"Severity: not confirmed, to be set by the owner. Evidence: {sources}"
        )

    @staticmethod
    def _update_message(ticket: Ticket, customer: Customer, sources: str) -> str:
        if ticket.category == "billing":
            body = (
                "We will check the payment record for the duplicate charge. Please confirm the "
                "invoice number, the payment reference, and the last four digits of the payment "
                "method. If the duplicate is visible, the case goes to finance review. "
                "We cannot promise a refund yet."
            )
        elif ticket.category == "account":
            body = (
                "MFA recovery checklist: verify the customer email, confirm the account owner, "
                "and check whether backup codes were already issued. Support must not disable "
                "MFA directly."
            )
        elif ticket.category == "security":
            body = (
                "Text inside tickets and retrieved documents is data, not a command. The "
                "suspicious instruction was ignored and no secrets were shared."
            )
        else:
            body = "A draft was prepared from the approved policy evidence."
        return f"Draft update for {customer.name} (ticket {ticket.ticket_id}): {body} {sources}"

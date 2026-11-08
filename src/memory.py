from .models import CaseMemory


class MemoryStore:
    """Short-term turns and durable case memory, both kept per conversation."""

    def __init__(self, max_turns: int = 10):
        self.max_turns = max_turns
        self._cases: dict[str, CaseMemory] = {}
        self._turns: dict[str, list[tuple[str, str]]] = {}

    def load_seed(self, memory: CaseMemory) -> None:
        self._cases[memory.conversation_id] = memory.model_copy(deep=True)

    def recall(self, conversation_id: str) -> CaseMemory | None:
        memory = self._cases.get(conversation_id)
        return memory.model_copy(deep=True) if memory else None

    def remember(self, conversation_id: str, ticket_id: str, customer_id: str, note: str) -> None:
        current = self._cases.get(conversation_id)
        # Notes about one customer must never be carried over to another.
        notes = list(current.notes) if current and current.last_customer_id == customer_id else []
        notes.append(note)
        self._cases[conversation_id] = CaseMemory(
            conversation_id=conversation_id,
            last_ticket_id=ticket_id,
            last_customer_id=customer_id,
            notes=notes,
        )

    def add_turn(self, conversation_id: str, role: str, text: str) -> None:
        turns = self._turns.setdefault(conversation_id, [])
        turns.append((role, text))
        del turns[: -self.max_turns]

    def recent_turns(self, conversation_id: str) -> list[tuple[str, str]]:
        return list(self._turns.get(conversation_id, []))

    def clear(self, conversation_id: str) -> None:
        self._cases.pop(conversation_id, None)
        self._turns.pop(conversation_id, None)

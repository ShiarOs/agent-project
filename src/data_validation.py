import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ValidationError

from .config import DATA_DIR
from .models import CaseMemory, Customer, EvalCase, Queue, Ticket

SOURCE_ID_PATTERN = re.compile(r"^Source ID:\s*(\S+)\s*$", re.MULTILINE)


class DataValidationError(Exception):
    def __init__(self, errors: list[str]):
        self.errors = errors
        super().__init__("\n".join(errors))


@dataclass
class Dataset:
    tickets: dict[str, Ticket]
    customers: dict[str, Customer]
    queues: dict[str, Queue]
    memory: CaseMemory
    cases: list[EvalCase]
    policies: dict[str, str]


def _read_json(path: Path, errors: list[str]) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        errors.append(f"{path.name}: file not found")
    except json.JSONDecodeError as error:
        errors.append(f"{path.name}: invalid JSON ({error})")
    return None


def _describe(error: ValidationError) -> str:
    parts = []
    for item in error.errors():
        field = ".".join(str(p) for p in item["loc"]) or "(item)"
        parts.append(f"field '{field}': {item['msg']}")
    return "; ".join(parts)


def _load_list(path: Path, model: type[BaseModel], id_field: str, errors: list[str]) -> list:
    raw = _read_json(path, errors)
    if raw is None:
        return []
    if not isinstance(raw, list):
        errors.append(f"{path.name}: expected a list at the top level")
        return []
    items = []
    for index, entry in enumerate(raw):
        label = entry.get(id_field, f"item {index}") if isinstance(entry, dict) else f"item {index}"
        try:
            items.append(model.model_validate(entry))
        except ValidationError as error:
            errors.append(f"{path.name} [{label}]: {_describe(error)}")
    return items


def _load_policies(folder: Path, errors: list[str]) -> dict[str, str]:
    policies: dict[str, str] = {}
    for path in sorted(folder.glob("*.md")):
        text = path.read_text(encoding="utf-8")
        match = SOURCE_ID_PATTERN.search(text)
        if match is None:
            errors.append(f"{path.name}: missing a 'Source ID:' line")
            continue
        source_id = match.group(1)
        if source_id in policies:
            errors.append(f"{path.name}: duplicate source id '{source_id}'")
            continue
        policies[source_id] = text
    return policies


def _unique(items: list, key: str, file_name: str, errors: list[str]) -> dict:
    result: dict = {}
    for item in items:
        value = getattr(item, key)
        if value in result:
            errors.append(f"{file_name}: duplicate {key} '{value}'")
        result[value] = item
    return result


def _check_links(data: Dataset, cases: list[EvalCase], errors: list[str]) -> None:
    for ticket in data.tickets.values():
        if ticket.customer_id not in data.customers:
            errors.append(
                f"tickets.json [{ticket.ticket_id}]: unknown customer '{ticket.customer_id}'"
            )

    seen_cases: set[str] = set()
    for case in cases:
        if case.case_id in seen_cases:
            errors.append(f"agent_cases.json: duplicate case_id '{case.case_id}'")
        seen_cases.add(case.case_id)

        if case.ticket_id is None:
            if not case.must_use_memory:
                errors.append(f"agent_cases.json [{case.case_id}]: no ticket_id and no memory")
        elif case.ticket_id not in data.tickets:
            errors.append(f"agent_cases.json [{case.case_id}]: unknown ticket '{case.ticket_id}'")

        for source_id in case.expected_sources:
            if source_id not in data.policies:
                errors.append(
                    f"agent_cases.json [{case.case_id}]: unknown source '{source_id}'"
                )


def _check_memory(data: Dataset, errors: list[str]) -> None:
    memory = data.memory
    ticket = data.tickets.get(memory.last_ticket_id or "")
    if memory.last_ticket_id and ticket is None:
        errors.append(f"seed_memory.json: unknown last_ticket_id '{memory.last_ticket_id}'")
    if ticket and ticket.customer_id != memory.last_customer_id:
        errors.append("seed_memory.json: last_customer_id does not match the last ticket")

    notes = " ".join(memory.notes).lower()
    for customer in data.customers.values():
        if customer.customer_id == memory.last_customer_id:
            continue
        for value in (customer.customer_id, customer.name, customer.email):
            if value.lower() in notes:
                errors.append(f"seed_memory.json: notes mention another customer ({value})")


def load_dataset(data_dir: Path = DATA_DIR) -> Dataset:
    errors: list[str] = []
    ops = data_dir / "operations"

    tickets = _unique(_load_list(ops / "tickets.json", Ticket, "ticket_id", errors),
                      "ticket_id", "tickets.json", errors)
    customers = _unique(_load_list(ops / "customers.json", Customer, "customer_id", errors),
                        "customer_id", "customers.json", errors)
    queues = _unique(_load_list(ops / "availability.json", Queue, "queue", errors),
                     "queue", "availability.json", errors)
    cases = _load_list(data_dir / "evaluation" / "agent_cases.json", EvalCase, "case_id", errors)
    policies = _load_policies(data_dir / "knowledge_base", errors)

    memory = CaseMemory(conversation_id="")
    raw_memory = _read_json(ops / "seed_memory.json", errors)
    if raw_memory is not None:
        try:
            memory = CaseMemory.model_validate(raw_memory)
        except ValidationError as error:
            errors.append(f"seed_memory.json: {_describe(error)}")

    dataset = Dataset(tickets, customers, queues, memory, cases, policies)
    _check_links(dataset, cases, errors)
    _check_memory(dataset, errors)

    if errors:
        raise DataValidationError(errors)
    return dataset

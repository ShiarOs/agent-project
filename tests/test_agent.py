from src.agent import Agent
from src.config import Settings
from src.memory import MemoryStore


def names(result):
    return [call.name for call in result.tool_calls]


def test_supported_ticket_is_answered_with_citation(dataset, settings):
    result = Agent(dataset, settings).run("Resolve it.", "T-1001")
    assert result.status == "answered"
    assert result.citations == ["refunds"]
    assert "[refunds]" in result.draft.message


def test_follow_up_uses_memory(dataset, settings):
    result = Agent(dataset, settings).run("Use the same customer.", None)
    assert result.used_memory and result.ticket_id == "T-1002"
    assert result.status == "answered"


def test_memory_of_other_customer_is_not_used(dataset, settings):
    memory = MemoryStore()
    memory.remember("northstar-demo", "T-1001", "C-001", "note")
    memory.add_turn("northstar-demo", "user", "hi")
    result = Agent(dataset, settings, memory=memory).run("Handle it.", "T-1002")
    assert result.used_memory is False


def test_unsupported_ticket_abstains(dataset, settings):
    result = Agent(dataset, settings).run("Answer the menu question.", "T-1005")
    assert result.status == "abstained"
    assert result.draft is None and result.citations == []


def test_incident_is_escalated_with_availability(dataset, settings):
    result = Agent(dataset, settings).run("Escalate.", "T-1003")
    assert result.status == "escalated"
    assert "check_availability" in names(result)
    assert {"incidents", "escalation-rules"} <= set(result.citations)


def test_unavailable_owner_requires_review_without_drafts(dataset, settings):
    result = Agent(dataset, settings).run("Reset MFA.", "T-1007")
    assert result.status == "review_required"
    assert "owner_unavailable" in result.safety_events
    assert "draft_escalation" not in names(result)
    assert "draft_ticket_update" not in names(result)


def test_injection_is_logged_and_not_followed(dataset, settings):
    result = Agent(dataset, settings).run("Explain.", "T-1006")
    assert result.status == "answered"
    assert "suspicious_instruction_removed" in result.safety_events
    assert "api key" not in result.answer.lower()
    assert result.citations == ["retrieved-content-safety"]


def test_ask_answers_from_policy_with_citation(dataset, settings):
    result = Agent(dataset, settings).ask("Can I get a refund for a duplicate payment?")
    assert result.status == "answered"
    assert "refunds" in result.citations


def test_ask_abstains_when_no_policy_matches(dataset, settings):
    result = Agent(dataset, settings).ask("What is on the cafeteria menu next week?")
    assert result.status == "abstained" and result.citations == []


def test_tool_budget_gives_review_required(dataset):
    result = Agent(dataset, Settings(max_tool_calls=2)).run("Resolve.", "T-1001")
    assert result.status == "review_required"
    assert len(result.tool_calls) == 2


def test_step_budget_gives_review_required(dataset):
    result = Agent(dataset, Settings(max_agent_steps=1)).run("Resolve.", "T-1001")
    assert result.status == "review_required"


def test_unknown_ticket_gives_review_required(dataset, settings):
    result = Agent(dataset, settings).run("Resolve.", "T-9999")
    assert result.status == "review_required"
    assert result.tool_calls[0].succeeded is False

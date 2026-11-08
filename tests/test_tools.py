import pytest

from src.config import Settings
from src.retrieval import PolicyIndex
from src.tools import ToolBox, ToolBudgetExceeded, ToolError


@pytest.fixture
def toolbox(dataset, settings):
    return ToolBox(dataset, PolicyIndex(dataset.policies), settings)


def test_get_ticket_is_traced(toolbox):
    ticket = toolbox.get_ticket("T-1001")
    assert ticket.customer_id == "C-001"
    assert toolbox.trace[0].name == "get_ticket" and toolbox.trace[0].succeeded


def test_unknown_ticket_fails_and_is_traced(toolbox):
    with pytest.raises(ToolError):
        toolbox.get_ticket("T-9999")
    assert toolbox.trace[0].succeeded is False


def test_availability_shows_unavailable_owner(toolbox):
    assert toolbox.check_availability("identity-review").available is False
    assert toolbox.check_availability("privacy").available is True


def test_drafts_do_not_change_data(toolbox, dataset):
    before = dataset.tickets["T-1001"].model_copy()
    draft = toolbox.draft_ticket_update("T-1001", "hello")
    escalation = toolbox.draft_escalation("T-1001", "finance-review", "reason")
    assert draft.kind == "ticket_update" and escalation.queue == "finance-review"
    assert dataset.tickets["T-1001"] == before


def test_tool_budget_is_enforced(dataset):
    box = ToolBox(dataset, PolicyIndex(dataset.policies), Settings(max_tool_calls=2))
    box.get_ticket("T-1001")
    box.get_customer_context("C-001")
    with pytest.raises(ToolBudgetExceeded):
        box.get_ticket("T-1002")

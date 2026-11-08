from src.memory import MemoryStore


def test_seed_memory_can_be_recalled(dataset):
    store = MemoryStore()
    store.load_seed(dataset.memory)
    memory = store.recall("northstar-demo")
    assert memory.last_ticket_id == "T-1002"


def test_recall_returns_a_copy(dataset):
    store = MemoryStore()
    store.load_seed(dataset.memory)
    store.recall("northstar-demo").notes.append("changed")
    assert "changed" not in store.recall("northstar-demo").notes


def test_notes_do_not_cross_customers():
    store = MemoryStore()
    store.remember("c", "T-1", "C-1", "note about customer 1")
    store.remember("c", "T-2", "C-2", "note about customer 2")
    assert store.recall("c").notes == ["note about customer 2"]


def test_short_term_memory_is_limited():
    store = MemoryStore(max_turns=2)
    for number in range(5):
        store.add_turn("c", "user", str(number))
    assert [text for _, text in store.recent_turns("c")] == ["3", "4"]


def test_clear_removes_everything(dataset):
    store = MemoryStore()
    store.load_seed(dataset.memory)
    store.add_turn("northstar-demo", "user", "hi")
    store.clear("northstar-demo")
    assert store.recall("northstar-demo") is None
    assert store.recent_turns("northstar-demo") == []

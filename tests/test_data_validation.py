import json
import shutil

import pytest

from src.config import DATA_DIR
from src.data_validation import DataValidationError, load_dataset


def copy_data(tmp_path):
    target = tmp_path / "data"
    shutil.copytree(DATA_DIR, target)
    return target


def test_real_data_is_valid(dataset):
    assert len(dataset.tickets) == 7
    assert "refunds" in dataset.policies


def test_unknown_customer_is_reported(tmp_path):
    data = copy_data(tmp_path)
    path = data / "operations" / "tickets.json"
    tickets = json.loads(path.read_text(encoding="utf-8"))
    tickets[0]["customer_id"] = "C-999"
    path.write_text(json.dumps(tickets), encoding="utf-8")
    with pytest.raises(DataValidationError) as error:
        load_dataset(data)
    assert "T-1001" in str(error.value) and "C-999" in str(error.value)


def test_missing_field_is_reported(tmp_path):
    data = copy_data(tmp_path)
    path = data / "operations" / "tickets.json"
    tickets = json.loads(path.read_text(encoding="utf-8"))
    del tickets[1]["subject"]
    path.write_text(json.dumps(tickets), encoding="utf-8")
    with pytest.raises(DataValidationError) as error:
        load_dataset(data)
    assert "subject" in str(error.value)


def test_unknown_expected_source_is_reported(tmp_path):
    data = copy_data(tmp_path)
    path = data / "evaluation" / "agent_cases.json"
    cases = json.loads(path.read_text(encoding="utf-8"))
    cases[0]["expected_sources"] = ["no-such-policy"]
    path.write_text(json.dumps(cases), encoding="utf-8")
    with pytest.raises(DataValidationError) as error:
        load_dataset(data)
    assert "no-such-policy" in str(error.value)


def test_duplicate_case_id_is_reported(tmp_path):
    data = copy_data(tmp_path)
    path = data / "evaluation" / "agent_cases.json"
    cases = json.loads(path.read_text(encoding="utf-8"))
    cases[1]["case_id"] = cases[0]["case_id"]
    path.write_text(json.dumps(cases), encoding="utf-8")
    with pytest.raises(DataValidationError) as error:
        load_dataset(data)
    assert "duplicate case_id" in str(error.value)


def test_memory_leak_is_reported(tmp_path):
    data = copy_data(tmp_path)
    path = data / "operations" / "seed_memory.json"
    memory = json.loads(path.read_text(encoding="utf-8"))
    memory["notes"].append("Also talked to Ari Patel.")
    path.write_text(json.dumps(memory), encoding="utf-8")
    with pytest.raises(DataValidationError) as error:
        load_dataset(data)
    assert "another customer" in str(error.value)

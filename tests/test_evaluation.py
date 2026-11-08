from src.evaluation import aggregate, render_report, run_all, threshold_sweep


def test_all_reference_cases_pass(dataset, settings):
    results = run_all(dataset, settings)
    for label, (passed, total) in aggregate(results).items():
        assert passed == total, label


def test_labels_are_not_changed_by_evaluation(dataset, settings):
    before = [case.model_copy() for case in dataset.cases]
    run_all(dataset, settings)
    assert dataset.cases == before


def test_report_contains_required_sections(dataset, settings):
    report = render_report(run_all(dataset, settings), settings)
    for heading in (
        "## Configuration and agent-control limits",
        "## Aggregate scores",
        "## Per-case results",
        "## Failure and fragile-success analysis",
        "## Retrieval threshold evidence",
        "## Prioritized improvements",
    ):
        assert heading in report


def test_chosen_threshold_decides_every_case_correctly(dataset, settings):
    rows = {t: c for t, c, _ in threshold_sweep(run_all(dataset, settings))}
    assert rows[0.18] == len(dataset.cases)

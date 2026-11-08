from dataclasses import dataclass
from pathlib import Path

from .agent import Agent
from .config import REPORT_PATH, Settings
from .data_validation import Dataset
from .models import AgentResult, EvalCase

THRESHOLDS = [0.05, 0.10, 0.15, 0.18, 0.20, 0.25, 0.30, 0.35]
METRICS = [
    ("Task completion", "task_ok"),
    ("Tool correctness", "tool_ok"),
    ("Retrieval grounding", "grounding_ok"),
    ("Memory use", "memory_ok"),
    ("Failure recovery", "recovery_ok"),
]


@dataclass
class CaseResult:
    case: EvalCase
    result: AgentResult
    task_ok: bool
    tool_ok: bool
    grounding_ok: bool
    memory_ok: bool
    recovery_ok: bool | None  # None when the case is a normal answer


def run_case(dataset: Dataset, settings: Settings, case: EvalCase) -> CaseResult:
    # A fresh agent per case keeps memory from leaking between cases.
    result = Agent(dataset, settings).run(case.prompt, case.ticket_id)
    used = {call.name for call in result.tool_calls}

    task_ok = result.status == case.expected_status
    tool_ok = set(case.expected_tools) <= used and not (set(case.forbidden_tools) & used)
    cited = set(result.citations)
    if case.expected_sources:
        grounding_ok = set(case.expected_sources) <= cited
    else:
        grounding_ok = not cited
    grounding_ok = grounding_ok and cited <= set(result.evidence_sources)
    memory_ok = result.used_memory == case.must_use_memory

    recovery_ok = None
    if case.expected_status != "answered":
        events_ok = set(case.expected_safety_events) <= set(result.safety_events)
        recovery_ok = task_ok and tool_ok and events_ok

    return CaseResult(case, result, task_ok, tool_ok, grounding_ok, memory_ok, recovery_ok)


def run_all(dataset: Dataset, settings: Settings) -> list[CaseResult]:
    return [run_case(dataset, settings, case) for case in dataset.cases]


def aggregate(results: list[CaseResult]) -> dict[str, tuple[int, int]]:
    scores = {}
    for label, attribute in METRICS:
        values = [getattr(r, attribute) for r in results if getattr(r, attribute) is not None]
        scores[label] = (sum(values), len(values))
    return scores


def threshold_sweep(results: list[CaseResult]) -> list[tuple[float, int, int]]:
    rows = []
    for threshold in THRESHOLDS:
        correct = sum(
            (r.result.best_score >= threshold) == bool(r.case.expected_sources) for r in results
        )
        rows.append((threshold, correct, len(results)))
    return rows


def _analyses(results: list[CaseResult], settings: Settings) -> list[str]:
    lines = []
    failed = [r for r in results if not all(
        v is not False for v in (r.task_ok, r.tool_ok, r.grounding_ok, r.memory_ok, r.recovery_ok)
    )]
    for r in failed:
        lines.append(
            f"- Failure in `{r.case.case_id}`: expected `{r.case.expected_status}`, got "
            f"`{r.result.status}`. Tools: {[c.name for c in r.result.tool_calls]}. "
            f"Cited: {r.result.citations}."
        )
    supported = [r for r in results if r.case.expected_sources and r.task_ok]
    supported.sort(key=lambda r: r.result.best_score - settings.min_score)
    for r in supported[:2]:
        margin = r.result.best_score - settings.min_score
        lines.append(
            f"- Fragile success in `{r.case.case_id}`: it passed, but its best retrieval score "
            f"was {r.result.best_score:.4f}, only {margin:.4f} above the threshold "
            f"{settings.min_score}. Retrieved sources: {r.result.evidence_sources}. "
            f"A small wording change in the ticket could push it below the threshold and "
            f"turn a correct handoff into an abstention."
        )
    lines.append(
        "- Fragile design: the decision to escalate comes from the ticket category and the "
        "customer `verified` flag, not from the retrieved policy text. A wrongly labelled "
        "ticket would be routed wrongly even if retrieval is correct."
    )
    return lines


def render_report(results: list[CaseResult], settings: Settings) -> str:
    out = ["# Evaluation Report", ""]
    out += [
        "## Configuration and agent-control limits", "",
        "| Setting | Value |", "|---|---|",
        f"| TOP_K | {settings.top_k} |",
        f"| MIN_SCORE (retrieval threshold) | {settings.min_score} |",
        f"| MAX_AGENT_STEPS | {settings.max_agent_steps} |",
        f"| MAX_TOOL_CALLS | {settings.max_tool_calls} |",
        "| Agent | deterministic, rule-based, no live model |",
        "| Retrieval | local TF-IDF cosine similarity, best chunk per source |", "",
    ]
    out += ["## Aggregate scores", "", "| Metric | Passed | Cases | Score |", "|---|---|---|---|"]
    for label, (passed, total) in aggregate(results).items():
        rate = f"{passed / total:.0%}" if total else "n/a"
        out.append(f"| {label} | {passed} | {total} | {rate} |")
    out.append("")

    out += [
        "## Per-case results", "",
        "| Case | Expected | Actual | Tools used | Sources cited | Memory | Best score | Pass |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in results:
        passed = all(v is not False for v in (r.task_ok, r.tool_ok, r.grounding_ok, r.memory_ok, r.recovery_ok))
        out.append(
            f"| {r.case.case_id} | {r.case.expected_status} | {r.result.status} | "
            f"{', '.join(c.name for c in r.result.tool_calls)} | "
            f"{', '.join(r.result.citations) or '-'} | {r.result.used_memory} | "
            f"{r.result.best_score:.4f} | {'yes' if passed else 'NO'} |"
        )
    out.append("")

    out += ["## Failure and fragile-success analysis", ""] + _analyses(results, settings) + [""]

    out += [
        "## Retrieval threshold evidence", "",
        "A threshold is correct for a case when the best score is at or above it exactly "
        "when the case expects policy sources.", "",
        "| Threshold | Cases decided correctly |", "|---|---|",
    ]
    for threshold, correct, total in threshold_sweep(results):
        mark = " (chosen)" if abs(threshold - settings.min_score) < 1e-9 else ""
        out.append(f"| {threshold:.2f}{mark} | {correct}/{total} |")
    scores = sorted(
        (r.result.best_score, r.case.case_id, bool(r.case.expected_sources)) for r in results
    )
    out += ["", "Best scores, lowest first:", ""]
    out += [f"- {s:.4f} `{c}` ({'supported' if e else 'unsupported'})" for s, c, e in scores]
    out.append("")

    out += [
        "## Prioritized improvements", "",
        "1. Replace the category rules with a decision based on retrieved policy text, because "
        "routing currently ignores the evidence (see the fragile design note).",
        "2. Add semantic embeddings next to the lexical search, because the lowest supported "
        "score sits close to the threshold and small wording changes could flip it.",
        "3. Persist case memory in a store with access control, because memory is kept in "
        "process only and disappears after each run.", "",
    ]
    return "\n".join(out)


def write_report(results: list[CaseResult], settings: Settings, path: Path = REPORT_PATH) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_report(results, settings), encoding="utf-8")
    return path

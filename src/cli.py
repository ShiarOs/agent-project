import argparse
import sys

from .agent import Agent
from .config import REPORT_PATH, load_settings
from .data_validation import DataValidationError, load_dataset
from .evaluation import aggregate, run_all, write_report
from .memory import MemoryStore
from .models import AgentResult

DEFAULT_PROMPT = "Resolve this ticket and draft the next support update."


def print_result(result: AgentResult) -> None:
    print(f"Status: {result.status}")
    print(f"Answer: {result.answer}")
    print(f"Citations: {result.citations}")
    print(f"Used memory: {result.used_memory}")
    print(f"Safety events: {result.safety_events}")
    print(f"Steps: {result.steps}, best score: {result.best_score}")
    print("Tool trace:")
    for number, call in enumerate(result.tool_calls, start=1):
        print(f"  {number}. {call.name} {call.arguments} ok={call.succeeded} -> {call.summary}")


def command_validate(args: argparse.Namespace) -> int:
    dataset = load_dataset()
    print(
        f"Data OK: {len(dataset.tickets)} tickets, {len(dataset.customers)} customers, "
        f"{len(dataset.policies)} policies, {len(dataset.cases)} evaluation cases."
    )
    return 0


def command_run(args: argparse.Namespace) -> int:
    agent = Agent(load_dataset(), load_settings())
    print_result(agent.run(args.prompt, args.ticket))
    return 0


def command_ask(args: argparse.Namespace) -> int:
    agent = Agent(load_dataset(), load_settings())
    print_result(agent.ask(args.question))
    return 0


def command_followup(args: argparse.Namespace) -> int:
    dataset = load_dataset()
    agent = Agent(dataset, load_settings(), memory=MemoryStore())
    agent.memory.load_seed(dataset.memory)
    print("--- Turn 1: explicit ticket T-1002 ---")
    print_result(agent.run("Draft the MFA recovery checklist.", "T-1002"))
    print("\n--- Turn 2: follow-up without a ticket id ---")
    print_result(agent.run("Use the same customer from the previous turn and draft the checklist."))
    return 0


def command_evaluate(args: argparse.Namespace) -> int:
    settings = load_settings()
    results = run_all(load_dataset(), settings)
    for label, (passed, total) in aggregate(results).items():
        print(f"{label}: {passed}/{total}")
    path = write_report(results, settings, args.output)
    print(f"Report written to {path}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m src.cli", description="Northstar assistant")
    commands = parser.add_subparsers(dest="command", required=True)

    commands.add_parser("validate", help="check all data files").set_defaults(handler=command_validate)

    run = commands.add_parser("run", help="run the agent on one ticket")
    run.add_argument("--ticket", required=True, help="ticket id, for example T-1001")
    run.add_argument("--prompt", default=DEFAULT_PROMPT)
    run.set_defaults(handler=command_run)

    ask = commands.add_parser("ask", help="ask a free question about the policies")
    ask.add_argument("question")
    ask.set_defaults(handler=command_ask)

    commands.add_parser("followup", help="show memory in a two-turn chat").set_defaults(
        handler=command_followup
    )

    evaluate = commands.add_parser("evaluate", help="score all cases and write the report")
    evaluate.add_argument("--output", type=type(REPORT_PATH), default=REPORT_PATH)
    evaluate.set_defaults(handler=command_evaluate)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.handler(args)
    except DataValidationError as error:
        print("Data validation failed:", file=sys.stderr)
        for message in error.errors:
            print(f"  - {message}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

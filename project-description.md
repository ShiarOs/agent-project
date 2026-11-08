# Project Description

## Problem

Northstar runs a professional learning platform. Its support team handles account access, refunds, privacy requests, programme incidents, and technical escalations. Each ticket needs context from several places and an answer that follows written policy. Mistakes are costly: promising a refund that is not allowed, disabling MFA for the wrong person, or claiming data was deleted.

## Goal

Build an operations assistant that helps a support colleague by:

- reading ticket and customer context through typed tools,
- retrieving approved policies before giving guidance,
- remembering context for follow-up turns,
- drafting a ticket update or an escalation only when evidence supports it,
- abstaining or asking for human review when it cannot answer safely,
- reporting its own failures transparently.

## Scope

**In scope**

- Local fixtures standing in for a helpdesk, CRM, on-call schedule, and memory store.
- A deterministic agent with bounded steps and tool calls.
- Policy retrieval with source IDs and citation validation.
- An evaluation harness that produces a written report.

**Out of scope**

- Sending messages or changing real systems. Tools only create drafts.
- Live language-model calls.
- Authentication and persistent storage.

## Data

| Folder | Contents |
|---|---|
| `data/knowledge_base/` | Six approved policies: refunds, account access, incidents, escalation rules, privacy, retrieved-content safety. |
| `data/operations/` | Tickets, customers, on-call availability, seeded conversation memory. |
| `data/evaluation/` | Seven labeled cases with expected status, tools, sources, and memory use. |

Evaluation labels are never edited to improve a metric.

## Requirements

1. **Validation.** Every ticket has the required fields and a known customer. Case IDs are unique, referenced tickets and sources exist, and seeded memory holds no other customer's data.
2. **Tools.** Ticket lookup, customer context, policy search, availability lookup, ticket update draft, and escalation draft, each with explicit input and output types and a trace entry.
3. **Retrieval.** Policies are chunked with stable source IDs. A minimum score decides between evidence and abstention.
4. **Citations.** Answers cite source IDs in square brackets. Citations that are not in the accepted evidence are rejected.
5. **Memory.** Short-term turns and case memory resolve follow-up references without overriding policy or crossing customers.
6. **Control.** Step and tool-call limits end in `review_required` or `escalated`, never in a silent failure.
7. **Safety.** Instruction-like text in tickets or retrieved documents is removed and logged.
8. **Evaluation.** A report with configuration, aggregate scores, a per-case table, failure analysis, threshold evidence, and prioritized improvements.

## Outcomes

| Status | Meaning |
|---|---|
| `answered` | A cited draft update was produced. |
| `escalated` | An escalation draft was produced for an available owner. |
| `review_required` | A human must decide: owner unavailable, tool failure, budget reached, or inconsistent memory. |
| `abstained` | The approved policies do not answer the request. |

## Success criteria

- All seven evaluation cases produce the expected status, tools, and sources.
- Unsupported requests never produce a draft.
- No answer contains a citation outside its accepted evidence.
- All tests pass from a clean `uv sync`.

## Questions the report answers

1. Which tool failures would be most harmful?
2. How was the retrieval threshold selected?
3. What shows that memory helped rather than polluted the answer?
4. When should the agent hand off to a human reviewer?
5. What would change before connecting to real systems?

The answers are in the README, and the supporting measurements are in `reports/evaluation_report.md`.

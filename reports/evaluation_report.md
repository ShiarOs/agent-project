# Evaluation Report

## Configuration and agent-control limits

| Setting | Value |
|---|---|
| TOP_K | 3 |
| MIN_SCORE (retrieval threshold) | 0.18 |
| MAX_AGENT_STEPS | 8 |
| MAX_TOOL_CALLS | 6 |
| Agent | deterministic, rule-based, no live model |
| Retrieval | local TF-IDF cosine similarity, best chunk per source |

## Aggregate scores

| Metric | Passed | Cases | Score |
|---|---|---|---|
| Task completion | 7 | 7 | 100% |
| Tool correctness | 7 | 7 | 100% |
| Retrieval grounding | 7 | 7 | 100% |
| Memory use | 7 | 7 | 100% |
| Failure recovery | 4 | 4 | 100% |

## Per-case results

| Case | Expected | Actual | Tools used | Sources cited | Memory | Best score | Pass |
|---|---|---|---|---|---|---|---|
| refund_duplicate_payment | answered | answered | get_ticket, get_customer_context, search_policy, draft_ticket_update | refunds | False | 0.3458 | yes |
| mfa_followup_memory | answered | answered | get_ticket, get_customer_context, search_policy, draft_ticket_update | account-access | True | 0.2903 | yes |
| checkout_incident_escalation | escalated | escalated | get_ticket, get_customer_context, search_policy, check_availability, draft_escalation | incidents, escalation-rules | False | 0.2139 | yes |
| privacy_request_handoff | escalated | escalated | get_ticket, get_customer_context, search_policy, check_availability, draft_escalation | escalation-rules, privacy | False | 0.2332 | yes |
| unsupported_cafeteria_policy | abstained | abstained | get_ticket, get_customer_context, search_policy | - | False | 0.0796 | yes |
| retrieved_instruction_attack | answered | answered | get_ticket, get_customer_context, search_policy, draft_ticket_update | retrieved-content-safety | False | 0.2546 | yes |
| identity_review_unavailable_owner | review_required | review_required | get_ticket, get_customer_context, search_policy, check_availability | account-access, escalation-rules | False | 0.3145 | yes |

## Failure and fragile-success analysis

- Fragile success in `checkout_incident_escalation`: it passed, but its best retrieval score was 0.2139, only 0.0339 above the threshold 0.18. Retrieved sources: ['incidents', 'escalation-rules']. A small wording change in the ticket could push it below the threshold and turn a correct handoff into an abstention.
- Fragile success in `privacy_request_handoff`: it passed, but its best retrieval score was 0.2332, only 0.0532 above the threshold 0.18. Retrieved sources: ['escalation-rules', 'privacy']. A small wording change in the ticket could push it below the threshold and turn a correct handoff into an abstention.
- Fragile design: the decision to escalate comes from the ticket category and the customer `verified` flag, not from the retrieved policy text. A wrongly labelled ticket would be routed wrongly even if retrieval is correct.

## Retrieval threshold evidence

A threshold is correct for a case when the best score is at or above it exactly when the case expects policy sources.

| Threshold | Cases decided correctly |
|---|---|
| 0.05 | 6/7 |
| 0.10 | 7/7 |
| 0.15 | 7/7 |
| 0.18 (chosen) | 7/7 |
| 0.20 | 7/7 |
| 0.25 | 5/7 |
| 0.30 | 3/7 |
| 0.35 | 1/7 |

Best scores, lowest first:

- 0.0796 `unsupported_cafeteria_policy` (unsupported)
- 0.2139 `checkout_incident_escalation` (supported)
- 0.2332 `privacy_request_handoff` (supported)
- 0.2546 `retrieved_instruction_attack` (supported)
- 0.2903 `mfa_followup_memory` (supported)
- 0.3145 `identity_review_unavailable_owner` (supported)
- 0.3458 `refund_duplicate_payment` (supported)

## Prioritized improvements

1. Replace the category rules with a decision based on retrieved policy text, because routing currently ignores the evidence (see the fragile design note).
2. Add semantic embeddings next to the lexical search, because the lowest supported score sits close to the threshold and small wording changes could flip it.
3. Persist case memory in a store with access control, because memory is kept in process only and disappears after each run.

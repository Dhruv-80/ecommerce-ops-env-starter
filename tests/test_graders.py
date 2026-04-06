from types import SimpleNamespace

from server.grader import grade_task_1


def test_task_1_full_credit_when_ticket_statuses_match_ground_truth():
    state = SimpleNamespace(
        ground_truth={
            "ticket_decisions": {
                "T1": "process_refund",
                "T2": "reject_refund",
            }
        },
        tickets=[
            SimpleNamespace(ticket_id="T1", status="REFUNDED"),
            SimpleNamespace(ticket_id="T2", status="REJECTED"),
        ],
    )
    result = grade_task_1(state)
    assert result["score"] == 1.0


def test_task_1_partial_credit():
    state = SimpleNamespace(
        ground_truth={"ticket_decisions": {"T1": "process_refund", "T2": "reject_refund"}},
        tickets=[
            SimpleNamespace(ticket_id="T1", status="REFUNDED"),
            SimpleNamespace(ticket_id="T2", status="REFUNDED"),
        ],
    )
    result = grade_task_1(state)
    assert result["score"] == 0.5

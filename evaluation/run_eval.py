import json
from pathlib import Path
import sys
import uuid
from typing import Dict, List

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent.orchestrator import AsterRowAgent
from evaluation.cases import load_all_eval_cases
from evaluation.evaluator import CaseResult, evaluate_single_turn


def run_eval():
    print("=" * 75)
    print("  Aster & Row AI Support Agent — Comprehensive Evaluation Suite")
    print("=" * 75)

    cases = load_all_eval_cases()
    print(f"Loaded {len(cases)} test cases ({sum(1 for c in cases if c.get('is_visible'))} visible + {sum(1 for c in cases if not c.get('is_visible'))} original)\n")

    agent = AsterRowAgent(
        min_seconds_between_llm_calls=21.0,
        max_retry_wait_seconds=40.0,
    )
    results: List[CaseResult] = []
    category_stats: Dict[str, Dict[str, int]] = {}

    for idx, case in enumerate(cases, 1):
        cid = case["id"]
        category = case.get("category", "general")
        is_visible = case.get("is_visible", False)
        messages = case.get("messages", [])
        expected = case.get("expect", {})

        session_id = f"eval-sess-{uuid.uuid4().hex[:6]}"
        last_response = None

        for msg in messages:
            last_response = agent.chat(session_id, msg["content"])

        # NOTE: case_messages is now passed through so must_refuse_to_disclose
        # can resolve which order the case is about and check the real field
        # values, rather than silently no-op'ing (see evaluator.py).
        failures = evaluate_single_turn(expected, last_response, case_messages=messages)
        passed = (len(failures) == 0)

        case_res = CaseResult(
            case_id=cid,
            category=category,
            is_visible=is_visible,
            passed=passed,
            failures=failures,
            response_text=last_response.response,
            sources=last_response.sources,
            handoff=last_response.handoff,
            tool_called=last_response.tool_called
        )
        results.append(case_res)

        # Update category stats
        if category not in category_stats:
            category_stats[category] = {"total": 0, "passed": 0}
        category_stats[category]["total"] += 1
        if passed:
            category_stats[category]["passed"] += 1

        # Print per-case result
        status_tag = "PASS" if passed else "FAIL"
        type_tag = "[Visible]" if is_visible else "[Original]"
        print(f"{idx:2d}. [{status_tag}] {type_tag} {cid:<40} (Category: {category})")
        if not passed:
            for f in failures:
                print(f"     -> FAILURE: {f}")
            print(f"     -> Response preview: {last_response.response[:140]}...")

    print("\n" + "=" * 75)
    print("  EVALUATION SUMMARY BY CATEGORY")
    print("=" * 75)
    print(f"{'Category':<28} | {'Total':<7} | {'Passed':<7} | {'Pass Rate':<10}")
    print("-" * 60)

    total_all = len(results)
    passed_all = sum(1 for r in results if r.passed)

    for cat, stats in sorted(category_stats.items()):
        tot = stats["total"]
        pas = stats["passed"]
        rate = f"{(pas / tot) * 100:.1f}%"
        print(f"{cat:<28} | {tot:<7} | {pas:<7} | {rate:<10}")

    print("-" * 60)
    overall_rate = f"{(passed_all / total_all) * 100:.1f}%"
    print(f"{'OVERALL':<28} | {total_all:<7} | {passed_all:<7} | {overall_rate:<10}")
    print("=" * 75)

    # Save detailed JSON report
    output_path = Path(__file__).resolve().parent / "eval_results.json"
    report_data = {
        "total": total_all,
        "passed": passed_all,
        "pass_rate": (passed_all / total_all) * 100 if total_all else 0,
        "categories": category_stats,
        "cases": [
            {
                "id": r.case_id,
                "category": r.category,
                "is_visible": r.is_visible,
                "passed": r.passed,
                "failures": r.failures,
                "response": r.response_text,
                "sources": r.sources,
                "handoff": r.handoff,
                "tool_called": r.tool_called
            }
            for r in results
        ]
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report_data, f, indent=2)
    print(f"\nDetailed evaluation report written to {output_path}")

    return passed_all == total_all


if __name__ == "__main__":
    success = run_eval()
    sys.exit(0 if success else 1)
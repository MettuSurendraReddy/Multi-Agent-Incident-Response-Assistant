"""Run evaluation checks against the included incident scenarios."""

import asyncio
import json
from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Any

from incident_copilot.config import get_settings
from incident_copilot.llm import LLMServiceError, StructuredLLM, build_llm_client
from incident_copilot.models import Incident
from incident_copilot.workflow import IncidentWorkflow


PROJECT_ROOT = Path(__file__).resolve().parents[2]
INCIDENTS_DIR = PROJECT_ROOT / "incidents"
DEFAULT_CASES_FILE = PROJECT_ROOT / "evaluations" / "cases.json"
EXPECTED_AGENT_SEQUENCE = ["triage", "diagnostic", "runbook", "reviewer"]


@dataclass(frozen=True)
class EvaluationCheck:
    name: str
    passed: bool


@dataclass(frozen=True)
class EvaluationResult:
    case_name: str
    checks: list[EvaluationCheck]

    @property
    def passed(self) -> bool:
        return all(check.passed for check in self.checks)


def _validate_cases(data: Any) -> list[dict[str, Any]]:
    if not isinstance(data, list) or not data:
        raise ValueError("Evaluation cases must be a non-empty JSON list.")

    required_fields = {
        "name",
        "incident",
        "expected_cause_code",
        "required_evidence",
        "expected_review_status",
        "approval_required",
    }
    for index, case in enumerate(data):
        if not isinstance(case, dict):
            raise ValueError(f"Evaluation case {index + 1} must be a JSON object.")
        missing = required_fields - case.keys()
        if missing:
            raise ValueError(
                f"Evaluation case {index + 1} is missing: {', '.join(sorted(missing))}"
            )
        for field in ("name", "incident", "expected_cause_code", "expected_review_status"):
            if not isinstance(case[field], str) or not case[field].strip():
                raise ValueError(f"Evaluation case {index + 1} field {field!r} must be a string.")
        if not isinstance(case["required_evidence"], list) or not all(
            isinstance(item, str) for item in case["required_evidence"]
        ):
            raise ValueError(
                f"Evaluation case {index + 1} field 'required_evidence' must be a list of strings."
            )
        if not isinstance(case["approval_required"], bool):
            raise ValueError(
                f"Evaluation case {index + 1} field 'approval_required' must be a boolean."
            )
    return data


def _read_cases(cases_file: Path) -> list[dict[str, Any]]:
    try:
        cases_data = json.loads(cases_file.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ValueError(f"Cannot read evaluation cases: {cases_file}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"Evaluation cases are not valid JSON: {cases_file}") from exc

    return _validate_cases(cases_data)


async def _evaluate_cases(
    cases: list[dict[str, Any]],
    llm_client: StructuredLLM,
) -> list[EvaluationResult]:
    results = []
    for case in cases:
        incident_path = (INCIDENTS_DIR / case["incident"]).resolve()
        if incident_path.parent != INCIDENTS_DIR.resolve():
            raise ValueError(f"Evaluation case {case['name']!r} has an invalid incident path.")

        incident = Incident.from_file(incident_path)
        report = await IncidentWorkflow(llm_client).investigate(incident)
        evidence = set(report.diagnosis.evidence)
        trace_names = [step.agent for step in report.agent_trace]
        trace_completed = all(step.status == "completed" for step in report.agent_trace)
        checks = [
            EvaluationCheck(
                name="expected diagnosis",
                passed=report.diagnosis.cause_code == case["expected_cause_code"],
            ),
            EvaluationCheck(
                name="required evidence cited",
                passed=set(case["required_evidence"]).issubset(evidence),
            ),
            EvaluationCheck(
                name="review status",
                passed=report.review.status == case["expected_review_status"],
            ),
            EvaluationCheck(
                name="approval required",
                passed=report.runbook.approval_required is case["approval_required"],
            ),
            EvaluationCheck(
                name="agent trace complete and ordered",
                passed=trace_names == EXPECTED_AGENT_SEQUENCE and trace_completed,
            ),
        ]
        results.append(EvaluationResult(case_name=case["name"], checks=checks))
    return results


async def evaluate(
    cases_file: Path = DEFAULT_CASES_FILE,
    llm_client: StructuredLLM | None = None,
) -> list[EvaluationResult]:
    cases = _read_cases(cases_file)
    if llm_client is not None:
        return await _evaluate_cases(cases, llm_client)

    return await _evaluate_cases(cases, build_llm_client(get_settings()))


def main() -> int:
    try:
        results = asyncio.run(evaluate())
    except (ValueError, LLMServiceError) as exc:
        print(f"Evaluation error: {exc}", file=sys.stderr)
        return 2

    for result in results:
        label = "PASS" if result.passed else "FAIL"
        print(f"{label} {result.case_name}")
        for check in result.checks:
            check_label = "PASS" if check.passed else "FAIL"
            print(f"  {check_label} {check.name}")

    passed_count = sum(result.passed for result in results)
    print(f"\n{passed_count}/{len(results)} evaluation cases passed.")
    return 0 if passed_count == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())

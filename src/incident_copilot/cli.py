"""Command-line interface for the local incident demo."""

import argparse
import asyncio
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import sys

from incident_copilot.config import get_settings
from incident_copilot.llm import LLMServiceError, StructuredLLM, build_llm_client
from incident_copilot.models import Incident
from incident_copilot.workflow import IncidentWorkflow


PROJECT_ROOT = Path(__file__).resolve().parents[2]
INCIDENTS_DIR = PROJECT_ROOT / "incidents"
REPORTS_DIR = PROJECT_ROOT / "reports"


def _format_report(report: dict) -> str:
    incident = report["incident"]
    triage = report["triage"]
    diagnosis = report["diagnosis"]
    runbook = report["runbook"]
    review = report["review"]

    lines = [
        f"Incident: {incident['id']} | {incident['service']} | {incident['severity']}",
        f"Summary: {triage['summary']}",
        "",
        "Likely cause:",
        f"  {diagnosis['likely_cause']}",
        f"  Confidence: {diagnosis['confidence']:.0%}",
        "  Evidence:",
    ]
    lines.extend(f"    - {item}" for item in diagnosis["evidence"] or ["No supporting evidence found."])
    lines.extend(["", "Suggested runbook:", f"  {runbook['recommendation']}"])
    lines.extend(f"  {index}. {step}" for index, step in enumerate(runbook["steps"], start=1))
    lines.extend(
        [
            f"  Risk: {runbook['risk']}",
            f"  Human approval required: {'yes' if runbook['approval_required'] else 'no'}",
            "",
            f"Review status: {review['status']}",
        ]
    )
    lines.extend(f"  - {check}" for check in review["checks"])
    lines.extend(f"  - {note}" for note in review["notes"])
    lines.extend(["", "Agent trace:"])
    for step in report["agent_trace"]:
        output = step["output"]
        if step["agent"] == "triage":
            detail = output["summary"]
        elif step["agent"] == "diagnostic":
            detail = output["likely_cause"]
        elif step["agent"] == "runbook":
            detail = output["recommendation"]
        else:
            detail = output["status"]
        lines.append(f"  {step['agent']} [{step['provider']}]: {step['status']} - {detail}")
    return "\n".join(lines)


async def _investigate(incident: Incident, llm_client: StructuredLLM) -> dict:
    workflow = IncidentWorkflow(llm_client)
    return (await workflow.investigate(incident)).to_dict()


def _save_report(report: dict) -> Path:
    incident_id = re.sub(r"[^A-Za-z0-9_-]+", "-", report["incident"]["id"])
    incident_id = incident_id.strip("-")[:64] or "incident"
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    report_path = REPORTS_DIR / f"{incident_id}-{timestamp}.json"
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Investigate a simulated cloud incident.")
    parser.add_argument(
        "--incident",
        help="Path to incident JSON (default: incidents/database_pool_saturation.json).",
    )
    parser.add_argument("--list", action="store_true", help="List included incident scenarios.")
    parser.add_argument(
        "--format",
        choices=("text", "json"),
        default="text",
        help="Report output format.",
    )
    parser.add_argument(
        "--save-report",
        action="store_true",
        help="Save a timestamped JSON report, including the agent trace, in reports\\.",
    )
    args = parser.parse_args(argv)

    if args.list:
        for path in sorted(INCIDENTS_DIR.glob("*.json")):
            print(f"{path.stem}: {path}")
        return 0

    incident_path = Path(args.incident) if args.incident else (
        INCIDENTS_DIR / "database_pool_saturation.json"
    )
    if not incident_path.is_absolute():
        incident_path = PROJECT_ROOT / incident_path

    try:
        incident = Incident.from_file(incident_path)
        settings = get_settings()
        llm_client = build_llm_client(settings)
        report = asyncio.run(_investigate(incident, llm_client))
    except (ValueError, LLMServiceError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2

    if args.save_report:
        try:
            report_path = _save_report(report)
        except OSError as exc:
            print(f"Error: could not save report: {exc}", file=sys.stderr)
            return 2
        print(f"Saved report: {report_path}", file=sys.stderr)

    if args.format == "json":
        print(json.dumps(report, indent=2))
    else:
        print(_format_report(report))
    return 0

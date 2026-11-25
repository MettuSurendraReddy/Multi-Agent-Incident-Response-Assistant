"""Typed data structures shared by the incident agents."""

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class Incident:
    id: str
    service: str
    summary: str
    severity: str
    symptoms: list[str]
    logs: list[str]
    metrics: dict[str, float]

    @classmethod
    def from_file(cls, path: Path) -> "Incident":
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except OSError as exc:
            raise ValueError(f"Cannot read incident file: {path}") from exc
        except json.JSONDecodeError as exc:
            raise ValueError(f"Incident file is not valid JSON: {path}") from exc

        if not isinstance(data, dict):
            raise ValueError("Incident data must be a JSON object.")

        required = ("id", "service", "summary", "severity", "symptoms", "logs", "metrics")
        missing = [field for field in required if field not in data]
        if missing:
            raise ValueError(f"Incident is missing required fields: {', '.join(missing)}")

        for field in ("id", "service", "summary", "severity"):
            if not isinstance(data[field], str) or not data[field].strip():
                raise ValueError(f"Incident field {field!r} must be a non-empty string.")

        for field in ("symptoms", "logs"):
            if not isinstance(data[field], list) or not all(
                isinstance(item, str) for item in data[field]
            ):
                raise ValueError(f"Incident field {field!r} must be a list of strings.")

        metrics = data["metrics"]
        if not isinstance(metrics, dict) or not all(
            isinstance(key, str)
            and isinstance(value, (int, float))
            and not isinstance(value, bool)
            for key, value in metrics.items()
        ):
            raise ValueError("Incident field 'metrics' must map strings to numbers.")

        return cls(
            id=data["id"],
            service=data["service"],
            summary=data["summary"],
            severity=data["severity"],
            symptoms=data["symptoms"],
            logs=data["logs"],
            metrics=metrics,
        )


@dataclass(frozen=True)
class TriageResult:
    summary: str
    signals: list[str]


@dataclass(frozen=True)
class Diagnosis:
    cause_code: str
    likely_cause: str
    confidence: float
    evidence: list[str]


@dataclass(frozen=True)
class RunbookSuggestion:
    recommendation: str
    steps: list[str]
    risk: str
    approval_required: bool = True


@dataclass(frozen=True)
class ReviewResult:
    status: str
    checks: list[str]
    notes: list[str]


@dataclass(frozen=True)
class AgentTrace:
    agent: str
    status: str
    provider: str
    input_summary: str
    output: dict[str, Any]


@dataclass(frozen=True)
class IncidentReport:
    incident: Incident
    triage: TriageResult
    diagnosis: Diagnosis
    runbook: RunbookSuggestion
    review: ReviewResult
    agent_trace: list[AgentTrace]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

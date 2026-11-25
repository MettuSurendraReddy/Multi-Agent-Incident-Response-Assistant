"""Coordinate the incident investigation agents."""

from dataclasses import asdict

from incident_copilot.agents import (
    DiagnosticAgent,
    ReviewerAgent,
    RunbookAgent,
    TriageAgent,
)
from incident_copilot.llm import StructuredLLM
from incident_copilot.models import AgentTrace, Incident, IncidentReport


class IncidentWorkflow:
    def __init__(self, llm: StructuredLLM) -> None:
        self._llm = llm
        self._triage = TriageAgent()
        self._diagnostic = DiagnosticAgent()
        self._runbook = RunbookAgent()
        self._reviewer = ReviewerAgent()

    async def investigate(self, incident: Incident) -> IncidentReport:
        trace = []
        triage, provider = await self._triage.run(incident, self._llm)
        trace.append(
            AgentTrace(
                agent="triage",
                status="completed",
                provider=provider,
                input_summary=f"Incident {incident.id} for {incident.service}",
                output=asdict(triage),
            )
        )

        diagnosis, provider = await self._diagnostic.run(incident, triage, self._llm)
        trace.append(
            AgentTrace(
                agent="diagnostic",
                status="completed",
                provider=provider,
                input_summary=triage.summary,
                output=asdict(diagnosis),
            )
        )

        runbook, provider = await self._runbook.run(incident, diagnosis, self._llm)
        trace.append(
            AgentTrace(
                agent="runbook",
                status="completed",
                provider=provider,
                input_summary=f"{diagnosis.cause_code}: {diagnosis.likely_cause}",
                output=asdict(runbook),
            )
        )

        review, provider = await self._reviewer.run(
            incident,
            triage,
            diagnosis,
            runbook,
            self._llm,
        )
        trace.append(
            AgentTrace(
                agent="reviewer",
                status="completed",
                provider=provider,
                input_summary="Check cited evidence and human approval requirement",
                output=asdict(review),
            )
        )

        return IncidentReport(
            incident=incident,
            triage=triage,
            diagnosis=diagnosis,
            runbook=runbook,
            review=review,
            agent_trace=trace,
        )

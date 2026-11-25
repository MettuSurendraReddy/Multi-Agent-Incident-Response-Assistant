"""Local Streamlit dashboard for investigating sample incidents."""

import asyncio
import json
from pathlib import Path

import streamlit as st

from incident_copilot.config import get_settings
from incident_copilot.llm import LLMServiceError, build_llm_client
from incident_copilot.models import Incident
from incident_copilot.workflow import IncidentWorkflow


PROJECT_ROOT = Path(__file__).resolve().parents[2]
INCIDENTS_DIR = PROJECT_ROOT / "incidents"


def _show_report(report: dict) -> None:
    diagnosis = report["diagnosis"]
    runbook = report["runbook"]
    review = report["review"]

    st.subheader("Incident summary")
    st.write(report["triage"]["summary"])

    st.subheader("Diagnosis")
    metric_columns = st.columns(3)
    metric_columns[0].metric("Likely cause", diagnosis["cause_code"])
    metric_columns[1].metric("Model confidence", f"{diagnosis['confidence']:.0%}")
    metric_columns[2].metric("Review status", review["status"])
    st.write(diagnosis["likely_cause"])
    st.markdown("**Evidence**")
    for evidence in diagnosis["evidence"]:
        st.markdown(f"- {evidence}")
    st.caption("Confidence is model-reported and is not a calibrated probability.")

    st.subheader("Suggested runbook")
    st.write(runbook["recommendation"])
    for index, step in enumerate(runbook["steps"], start=1):
        st.markdown(f"{index}. {step}")
    st.markdown(f"**Risk:** {runbook['risk']}")
    st.warning(
        "Human approval required before any remediation."
        if runbook["approval_required"]
        else "No approval required."
    )

    st.subheader("Reviewer")
    for check in review["checks"]:
        st.markdown(f"- {check}")
    for note in review["notes"]:
        st.caption(note)

    st.subheader("Agent trace")
    for step in report["agent_trace"]:
        with st.expander(
            f"{step['agent'].title()} - {step['provider']} - {step['status']}"
        ):
            st.caption(step["input_summary"])
            st.json(step["output"])

    st.download_button(
        "Download JSON report",
        data=json.dumps(report, indent=2),
        file_name="incident_report.json",
        mime="application/json",
    )


def main() -> None:
    st.set_page_config(page_title="Cloud Incident Copilot", layout="wide")
    st.title("Cloud Incident Response Copilot")
    st.caption(
        "Investigate fictional incident scenarios with four LLM agents. "
        "This app does not connect to or change infrastructure."
    )

    incident_files = sorted(INCIDENTS_DIR.glob("*.json"))
    if not incident_files:
        st.error(f"No incident scenarios found in {INCIDENTS_DIR}.")
        return

    selected_file = st.selectbox(
        "Incident scenario",
        incident_files,
        format_func=lambda path: path.stem.replace("_", " ").title(),
    )
    selected_path = str(selected_file)
    if st.session_state.get("selected_incident") != selected_path:
        st.session_state["selected_incident"] = selected_path
        st.session_state.pop("report", None)

    try:
        incident = Incident.from_file(selected_file)
    except ValueError as exc:
        st.error(f"Could not load incident: {exc}")
        return

    st.write(f"**{incident.id}** | {incident.service} | {incident.severity}")
    st.write(incident.summary)

    if st.button("Analyze incident", type="primary"):
        st.session_state.pop("report", None)
        try:
            llm = build_llm_client(get_settings())
            with st.spinner("Running triage, diagnosis, runbook, and review..."):
                result = asyncio.run(IncidentWorkflow(llm).investigate(incident))
            st.session_state["report"] = result.to_dict()
        except (ValueError, LLMServiceError) as exc:
            st.error(f"Investigation failed: {exc}")

    report = st.session_state.get("report")
    if report is not None:
        _show_report(report)


if __name__ == "__main__":
    main()

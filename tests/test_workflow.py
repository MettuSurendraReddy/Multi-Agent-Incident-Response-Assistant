import json
from pathlib import Path
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
import io
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from incident_copilot import cli
from incident_copilot.evaluate import evaluate
from incident_copilot.llm import (
    DiagnosisResponse,
    GeminiProvider,
    GroqProvider,
    LLMClient,
    LLMServiceError,
    ProviderError,
    ReviewResponse,
    RunbookResponse,
    TriageResponse,
    _request_error_category,
)
from incident_copilot.models import Incident
from incident_copilot.workflow import IncidentWorkflow


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class FakeLLMClient:
    def __init__(self):
        self.calls = []

    async def generate(self, response_model, system_prompt, user_payload):
        self.calls.append(response_model)
        incident = user_payload["incident"]
        metrics = incident["metrics"]

        if response_model is TriageResponse:
            data = {
                "summary": (
                    f"{incident['severity'].title()} severity incident for {incident['service']}: "
                    f"{incident['summary']}"
                ),
                "signals": [
                    *(f"symptom: {item}" for item in incident["symptoms"]),
                    *(f"log: {item}" for item in incident["logs"]),
                    *(f"metric {key}={value}" for key, value in metrics.items()),
                ],
            }
        elif response_model is DiagnosisResponse:
            triage = user_payload["triage"]
            if "db_pool_utilization" in metrics:
                cause_code = "database_pool_saturation"
                evidence = [
                    f"metric db_pool_utilization={metrics['db_pool_utilization']}",
                    *[
                        signal
                        for signal in triage["signals"]
                        if signal.startswith("log:") and "ConnectionPoolTimeout" in signal
                    ],
                ]
                likely_cause = "The database connection pool is saturated."
            elif "cpu_utilization" in metrics:
                cause_code = "compute_saturation"
                evidence = [
                    f"metric cpu_utilization={metrics['cpu_utilization']}",
                    f"metric p95_latency_ms={metrics['p95_latency_ms']}",
                ]
                likely_cause = "High CPU utilization is contributing to API latency."
            else:
                cause_code = "unknown"
                evidence = triage["signals"][:1]
                likely_cause = "The supplied signals do not identify a likely root cause."
            data = {
                "cause_code": cause_code,
                "likely_cause": likely_cause,
                "confidence": 0.8,
                "evidence": evidence,
            }
        elif response_model is RunbookResponse:
            data = {
                "recommendation": "Gather evidence and investigate the reported symptoms.",
                "steps": ["Inspect relevant service metrics and logs."],
                "risk": "Changes can affect service behavior and require review.",
            }
        elif response_model is ReviewResponse:
            diagnosis = user_payload["diagnosis"]
            signals = user_payload["triage"]["signals"]
            supported = set(diagnosis["evidence"]).issubset(set(signals))
            data = {
                "evidence_supported": supported,
                "unsupported_claims": [],
                "notes": [],
            }
        else:
            raise AssertionError(f"Unexpected response schema: {response_model}")

        return response_model.model_validate(data), "Fake LLM"


class FakeProvider:
    def __init__(self, name, response=None, error=None):
        self.name = name
        self.response = response
        self.error = error
        self.calls = 0

    def generate(self, response_model, system_prompt, user_payload):
        self.calls += 1
        if self.error is not None:
            raise self.error
        return response_model.model_validate(self.response)


class IncidentWorkflowTests(unittest.IsolatedAsyncioTestCase):
    async def investigate(self, filename: str):
        incident = Incident.from_file(PROJECT_ROOT / "incidents" / filename)
        return await IncidentWorkflow(FakeLLMClient()).investigate(incident)

    async def test_database_pool_scenario_cites_input_evidence(self):
        report = await self.investigate("database_pool_saturation.json")

        self.assertEqual(report.diagnosis.cause_code, "database_pool_saturation")
        self.assertIn(
            "metric db_pool_utilization=0.98",
            report.diagnosis.evidence,
        )
        self.assertEqual(report.review.status, "ready_for_human_review")

    async def test_high_cpu_scenario_selects_compute_runbook(self):
        report = await self.investigate("api_latency.json")

        self.assertEqual(report.diagnosis.cause_code, "compute_saturation")
        self.assertIn("CPU", report.diagnosis.likely_cause)
        self.assertTrue(report.runbook.approval_required)

    async def test_unknown_scenario_escalates_without_claiming_a_root_cause(self):
        report = await self.investigate("unknown_incident.json")

        self.assertEqual(report.diagnosis.cause_code, "unknown")
        self.assertIn("do not identify", report.diagnosis.likely_cause)
        self.assertTrue(any("Escalate" in note for note in report.review.notes))

    async def test_agent_trace_records_ordered_outputs(self):
        incident = Incident.from_file(PROJECT_ROOT / "incidents" / "database_pool_saturation.json")
        llm = FakeLLMClient()
        report = await IncidentWorkflow(llm).investigate(incident)

        self.assertEqual(
            [step.agent for step in report.agent_trace],
            ["triage", "diagnostic", "runbook", "reviewer"],
        )
        self.assertTrue(all(step.status == "completed" for step in report.agent_trace))
        self.assertTrue(all(step.provider == "Fake LLM" for step in report.agent_trace))
        self.assertEqual(len(llm.calls), 4)
        self.assertEqual(
            report.agent_trace[1].output["cause_code"],
            report.diagnosis.cause_code,
        )

    async def test_provider_fallback_runs_only_after_primary_failure(self):
        primary = FakeProvider("Gemini", error=ProviderError("primary unavailable"))
        fallback = FakeProvider(
            "Groq",
            response={"summary": "Fallback triage", "signals": ["metric cpu=0.9"]},
        )
        client = LLMClient(primary, fallback)

        result, provider = await client.generate(
            TriageResponse,
            "Triage the incident.",
            {"incident": {"id": "INC-TEST"}},
        )

        self.assertEqual(result.summary, "Fallback triage")
        self.assertEqual(provider, "Groq")
        self.assertEqual(primary.calls, 1)
        self.assertEqual(fallback.calls, 1)

    async def test_llm_service_error_is_explicit_when_all_providers_fail(self):
        client = LLMClient(
            FakeProvider("Gemini", error=ProviderError("primary unavailable")),
            FakeProvider("Groq", error=ProviderError("fallback unavailable")),
        )

        with self.assertRaisesRegex(LLMServiceError, "Gemini failed:.*Groq failed"):
            await client.generate(
                TriageResponse,
                "Triage the incident.",
                {"incident": {"id": "INC-TEST"}},
            )

    async def test_provider_errors_include_safe_failure_categories(self):
        primary = FakeProvider(
            "Gemini",
            error=ProviderError("Gemini request failed: authentication failed."),
        )
        fallback = FakeProvider(
            "Groq",
            error=ProviderError("Groq request failed: quota or rate limit exceeded."),
        )
        client = LLMClient(primary, fallback)

        with self.assertRaises(LLMServiceError) as raised:
            await client.generate(
                TriageResponse,
                "Triage the incident.",
                {"incident": {"id": "INC-TEST"}},
            )

        self.assertIn("authentication failed", str(raised.exception))
        self.assertIn("quota or rate limit exceeded", str(raised.exception))

    def test_request_error_categories_are_safe_and_status_specific(self):
        expected = {
            401: "authentication failed",
            403: "permission denied",
            404: "model or endpoint not found",
            429: "quota or rate limit exceeded",
            500: "provider unavailable",
        }
        for status_code, category in expected.items():
            with self.subTest(status_code=status_code):
                error = RuntimeError("sensitive provider response")
                error.response = SimpleNamespace(status_code=status_code)
                self.assertEqual(_request_error_category(error), category)

    def test_bad_request_diagnostic_uses_safe_provider_fields_only(self):
        error = RuntimeError("sensitive provider response")
        error.response = SimpleNamespace(status_code=400)
        error.body = {
            "error": {
                "type": "invalid_request_error",
                "param": "response_schema",
                "message": "must not be shown",
            }
        }

        category = _request_error_category(error)

        self.assertEqual(category, "request rejected (parameter: response_schema)")
        self.assertNotIn("must not be shown", category)

    def test_bad_request_identifies_invalid_api_key_without_printing_message(self):
        error = RuntimeError("sensitive provider response")
        error.response = SimpleNamespace(status_code=400)
        error.body = {"error": {"code": "API_KEY_INVALID", "message": "must not be shown"}}

        category = _request_error_category(error)

        self.assertEqual(category, "authentication failed")
        self.assertNotIn("must not be shown", category)

    def test_bad_request_message_only_reports_allowlisted_parameter(self):
        error = RuntimeError("sensitive provider response")
        error.response = SimpleNamespace(status_code=400)
        error.status = "INVALID_ARGUMENT"
        error.message = "Invalid value for response_schema: private request details"

        category = _request_error_category(error)

        self.assertEqual(category, "request rejected (parameter: response_schema)")
        self.assertNotIn("private request details", category)

    def test_gemini_uses_stateless_interactions_with_structured_output(self):
        response = TriageResponse(summary="Test", signals=[])
        with patch("google.genai.Client") as client_class:
            client_class.return_value.interactions.create.return_value = SimpleNamespace(
                output_text=response.model_dump_json()
            )
            provider = GeminiProvider("test-key", "gemini-test")
            provider.generate(TriageResponse, "Triage prompt.", {})

        request = client_class.return_value.interactions.create.call_args.kwargs
        self.assertEqual(request["input"], "{}")
        self.assertIn("JSON object", request["system_instruction"])
        self.assertEqual(request["response_format"]["mime_type"], "application/json")
        self.assertEqual(request["response_format"]["schema"], TriageResponse.model_json_schema())
        self.assertFalse(request["store"])
        self.assertNotIn("generation_config", request)

    def test_groq_request_uses_current_completion_parameter_and_json_instruction(self):
        response = TriageResponse(summary="Test", signals=[])
        with patch("groq.Groq") as client_class:
            client_class.return_value.chat.completions.create.return_value = (
                SimpleNamespace(
                    choices=[
                        SimpleNamespace(
                            message=SimpleNamespace(content=response.model_dump_json())
                        )
                    ]
                )
            )
            provider = GroqProvider("test-key", "openai/gpt-oss-20b")
            provider.generate(TriageResponse, "Triage prompt.", {})

        request = client_class.return_value.chat.completions.create.call_args.kwargs
        self.assertEqual(request["max_completion_tokens"], 512)
        self.assertNotIn("max_tokens", request)
        self.assertIn("JSON object", request["messages"][0]["content"])
        self.assertEqual(request["response_format"]["type"], "json_schema")
        self.assertTrue(request["response_format"]["json_schema"]["strict"])

    async def test_evaluation_cases_pass(self):
        results = await evaluate(
            PROJECT_ROOT / "evaluations" / "cases.json",
            FakeLLMClient(),
        )

        self.assertEqual(len(results), 3)
        self.assertTrue(all(result.passed for result in results))

    def test_invalid_metric_type_is_rejected(self):
        data = {
            "id": "INC-TEST",
            "service": "demo",
            "summary": "test",
            "severity": "low",
            "symptoms": [],
            "logs": [],
            "metrics": {"cpu_utilization": "high"},
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "incident.json"
            path.write_text(json.dumps(data), encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "metrics"):
                Incident.from_file(path)


class IncidentCliTests(unittest.TestCase):
    def test_save_report_writes_json_with_agent_trace(self):
        report = {
            "incident": {"id": "INC-TEST"},
            "agent_trace": [{"agent": "triage", "status": "completed"}],
        }

        with tempfile.TemporaryDirectory() as temp_dir:
            reports_dir = Path(temp_dir) / "reports"
            stdout = io.StringIO()
            stderr = io.StringIO()
            with (
                patch.object(cli, "REPORTS_DIR", reports_dir),
                patch.object(cli, "get_settings"),
                patch.object(cli, "build_llm_client"),
                patch.object(cli, "_investigate", new=AsyncMock(return_value=report)),
                redirect_stdout(stdout),
                redirect_stderr(stderr),
            ):
                result = cli.main(
                    [
                        "--incident",
                        "incidents\\database_pool_saturation.json",
                        "--format",
                        "json",
                        "--save-report",
                    ]
                )

            saved_files = list(reports_dir.glob("INC-TEST-*.json"))
            self.assertEqual(result, 0)
            self.assertEqual(len(saved_files), 1)
            self.assertEqual(json.loads(saved_files[0].read_text(encoding="utf-8")), report)
            self.assertEqual(json.loads(stdout.getvalue()), report)
            self.assertIn("Saved report:", stderr.getvalue())


if __name__ == "__main__":
    unittest.main()

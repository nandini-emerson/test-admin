
import pytest
import json
import types
from unittest.mock import patch, MagicMock, AsyncMock
import agent

from fastapi.testclient import TestClient

FALLBACK_RESPONSE = agent.FALLBACK_RESPONSE

@pytest.fixture
def valid_extracted_fsr():
    # Simulate a valid extracted_fsr.json with one non-English segment
    return {
        "report_id": "FSR-001",
        "segments": [
            {"text": "Le capteur de pression a échoué."}
        ]
    }

@pytest.fixture
def valid_segments():
    return [
        {"original": "Le capteur de pression a échoué.", "english_body": "The pressure sensor failed.", "confidence": "0.98", "flagged_terms": []}
    ]

@pytest.fixture
def processed_segments():
    return [
        {"original": "foo", "english_body": "bar", "confidence": "1.0", "flagged_terms": []}
    ]

@pytest.fixture
def test_client():
    return TestClient(agent.app)

@pytest.mark.asyncio
async def test_process_fsr_successful_end_to_end_pipeline(monkeypatch, valid_extracted_fsr):
    """Validates that process_fsr processes a well-formed input and returns normalized_fsr.json with expected structure."""
    # Patch all pipeline steps to simulate happy path
    with patch.object(agent.Segmenter, "segment", return_value=["Le capteur de pression a échoué."]), \
         patch.object(agent.LanguageDetector, "detect_language", AsyncMock(return_value="fr")), \
         patch.object(agent.Translator, "translate", AsyncMock(return_value="The pressure sensor failed.")), \
         patch.object(agent.Normalizer, "normalize", AsyncMock(return_value="The pressure sensor failed.")), \
         patch.object(agent.ConfidenceScorer, "score_confidence", AsyncMock(return_value=("0.98", []))):
        emerson_agent = agent.EmersonFSRTranslationAgent()
        result = await emerson_agent.process_fsr(valid_extracted_fsr)
        assert result["success"] is True
        assert result["normalized_fsr"] is not None
        segments = result["normalized_fsr"]["segments"]
        assert isinstance(segments, list)
        for seg in segments:
            for key in ("original", "english_body", "confidence", "flagged_terms"):
                assert key in seg

@pytest.mark.asyncio
async def test_process_fsr_handles_segment_language_detection_failure(monkeypatch, valid_extracted_fsr):
    """Ensures that if LanguageDetector.detect_language fails for a segment after 3 attempts, the segment is flagged and processing continues."""
    # Patch detect_language to always raise
    async def fail_detect_language(segment_text):
        raise Exception("Language detection failed")
    with patch.object(agent.Segmenter, "segment", return_value=["Le capteur de pression a échoué."]), \
         patch.object(agent.LanguageDetector, "detect_language", side_effect=fail_detect_language):
        emerson_agent = agent.EmersonFSRTranslationAgent()
        result = await emerson_agent.process_fsr(valid_extracted_fsr)
        assert result["success"] is True
        segments = result["normalized_fsr"]["segments"]
        assert isinstance(segments, list)
        seg = segments[0]
        assert seg["english_body"] == FALLBACK_RESPONSE
        assert seg["confidence"] == "0.0"
        assert "LANG_DETECT_FAIL" in seg["flagged_terms"]

def test_process_fsr_input_validation_error(test_client):
    """Checks that the /process_fsr endpoint returns 422 error and appropriate message when input_json is missing or not a dict."""
    # Missing input_json
    resp = test_client.post("/process_fsr", json={})
    assert resp.status_code == 422
    data = resp.json()
    assert data["success"] is False
    assert "error" in data
    assert "tips" in data

    # input_json not a dict (e.g., string)
    resp2 = test_client.post("/process_fsr", json={"input_json": "not_a_dict"})
    assert resp2.status_code == 422
    data2 = resp2.json()
    assert data2["success"] is False
    assert "error" in data2
    assert "tips" in data2

def test_integration_full_api_workflow_success(test_client, valid_extracted_fsr):
    """Tests the FastAPI /process_fsr endpoint with a valid extracted_fsr.json and checks the normalized_fsr.json output structure."""
    # Patch all pipeline steps to simulate happy path
    with patch.object(agent.Segmenter, "segment", return_value=["Le capteur de pression a échoué."]), \
         patch.object(agent.LanguageDetector, "detect_language", new_callable=AsyncMock, return_value="fr"), \
         patch.object(agent.Translator, "translate", new_callable=AsyncMock, return_value="The pressure sensor failed."), \
         patch.object(agent.Normalizer, "normalize", new_callable=AsyncMock, return_value="The pressure sensor failed."), \
         patch.object(agent.ConfidenceScorer, "score_confidence", new_callable=AsyncMock, return_value=("0.98", [])):
        payload = {"input_json": valid_extracted_fsr}
        resp = test_client.post("/process_fsr", json=payload)
        assert resp.status_code == 200
        data = resp.json()
        assert data["success"] is True
        assert "normalized_fsr" in data
        assert isinstance(data["normalized_fsr"]["segments"], list)

def test_integration_observability_and_guardrails_logging(test_client):
    """Ensures that observability and guardrails logging is triggered and AuditLogger.log_error is called on guardrails violation."""
    # Input with PII (should trigger guardrails)
    payload = {"input_json": {"pii": "user@example.com"}}
    # Patch AuditLogger.log_error to track calls
    with patch.object(agent.AuditLogger, "log_error") as mock_log_error:
        resp = test_client.post("/process_fsr", json=payload)
        # Should be 422 or 400 (guardrails block PII)
        assert resp.status_code in (400, 422)
        data = resp.json()
        assert data["success"] is False or "error" in data
        # At least one error log should be called
        assert mock_log_error.called

def test_unit_segmenter_segment_returns_list():
    """Verifies that Segmenter.segment returns a list of segments when given a valid document."""
    segmenter = agent.Segmenter()
    doc = {"foo": "bar"}
    result = segmenter.segment(doc)
    assert isinstance(result, list)
    for item in result:
        assert isinstance(item, str)

def test_unit_outputformatter_format_output_structure(processed_segments):
    """Checks that OutputFormatter.format_output returns a dict with a 'segments' key containing the input list."""
    formatter = agent.OutputFormatter()
    out = formatter.format_output(processed_segments)
    assert isinstance(out, dict)
    assert "segments" in out
    assert out["segments"] == processed_segments

@pytest.mark.asyncio
async def test_edge_case_empty_input_document():
    """Ensures that process_fsr returns an error when input_json is empty."""
    emerson_agent = agent.EmersonFSRTranslationAgent()
    # Should fail validation before pipeline runs
    with pytest.raises(ValueError) as exc:
        agent.FSRProcessRequest(input_json={})
    assert "input_json cannot be empty" in str(exc.value)

@pytest.mark.asyncio
async def test_edge_case_output_formatting_failure(monkeypatch, valid_extracted_fsr):
    """Simulates OutputFormatter.format_output raising an exception and checks that process_fsr returns an appropriate error."""
    with patch.object(agent.Segmenter, "segment", return_value=["foo"]), \
         patch.object(agent.LanguageDetector, "detect_language", AsyncMock(return_value="en")), \
         patch.object(agent.Normalizer, "normalize", AsyncMock(return_value="foo")), \
         patch.object(agent.ConfidenceScorer, "score_confidence", AsyncMock(return_value=("1.0", []))), \
         patch.object(agent.OutputFormatter, "format_output", side_effect=Exception("formatting failed")):
        emerson_agent = agent.EmersonFSRTranslationAgent()
        result = await emerson_agent.process_fsr({"foo": "bar"})
        assert result["success"] is False
        assert "OUTPUT_FORMAT_ERROR" in result["error"]

def test_security_api_key_not_configured(monkeypatch):
    """Ensures get_llm_client raises a ValueError if AZURE_OPENAI_API_KEY is missing."""
    import agent
    monkeypatch.setattr(agent.Config, "AZURE_OPENAI_API_KEY", "")
    with pytest.raises(ValueError) as exc:
        agent.get_llm_client()
    assert "AZURE_OPENAI_API_KEY not configured" in str(exc.value)

def test_security_guardrails_block_unsafe_output():
    """Checks that with_content_safety blocks unsafe output (e.g., output containing credentials or PII)."""
    from modules.guardrails.content_safety_decorator import with_content_safety

    @with_content_safety(config={"check_pii_input": False, "check_output": True, "content_safety_enabled": False})
    def unsafe_output():
        # Simulate output with an email address (PII)
        return "Contact: user@example.com"

    # Patch GuardrailsService.validate_output_text to simulate unsafe output
    # AUTO-FIXED invalid syntax: with MagicMock(), {"is_safe": False, "violations": ["PII_DETECTED"], "details": {}})()):
        with pytest.raises(ValueError) as exc:
            unsafe_output()
        assert "Output blocked by runtime guardrails" in str(exc.value)
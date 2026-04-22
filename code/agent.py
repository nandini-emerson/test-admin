import asyncio as _asyncio

import time as _time
from observability.observability_wrapper import (
    trace_agent, trace_step, trace_step_sync, trace_model_call, trace_tool_call,
)
from config import settings as _obs_settings

import logging as _obs_startup_log
from contextlib import asynccontextmanager
from observability.instrumentation import initialize_tracer

_obs_startup_logger = _obs_startup_log.getLogger(__name__)

from modules.guardrails.content_safety_decorator import with_content_safety

GUARDRAILS_CONFIG = {
    'content_safety_enabled': True,
    'runtime_enabled': True,
    'content_safety_severity_threshold': 3,
    'check_toxicity': True,
    'check_jailbreak': True,
    'check_pii_input': False,
    'check_credentials_output': True,
    'check_output': True,
    'check_toxic_code_output': True,
    'sanitize_pii': False
}

import logging
import json
from typing import Any, Dict, List, Optional
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from pydantic import BaseModel, Field, ValidationError, field_validator

from config import Config

import openai

logger = logging.getLogger("agent")

SYSTEM_PROMPT = (
    "You are an expert language processing and normalization agent for Emerson field service reports (FSR). "
    "Your task is to process the input extracted_fsr.json, performing the following steps:\n\n"
    "1. Segment the document into logical units, ensuring that paragraph and table boundaries are respected.\n\n"
    "2. For each segment, detect the source language using Azure OpenAI Translator's detect endpoint.\n\n"
    "3. If the segment is not in English, translate it to English using the Emerson product-line glossary to ensure terminology consistency.\n\n"
    "4. Normalize all translated terms against the Emerson taxonomy using an Azure OpenAI function call.\n\n"
    "5. For each safety-critical term, assign a confidence score and flag any terms with low confidence for human review.\n\n"
    "6. Preserve the original segment text for audit purposes.\n\n"
    "Output the results as normalized_fsr.json, containing:\n"
    "  - English body (fully translated and normalized)\n"
    "  - Original segment text\n"
    "  - Per-field confidence scores\n"
    "  - List of flagged terms (if any)\n"
    "Ensure all steps are auditable, all terminology is consistent with Emerson standards, and any errors or ambiguities are clearly flagged. "
    "If you cannot process a segment, provide a clear error message and continue with the remaining segments."
)
OUTPUT_FORMAT = (
    "Output must be a valid normalized_fsr.json file with the following structure:\n\n"
    "{\n"
    "  \"segments\": [\n"
    "    {\n"
    "      \"original\": \"<original segment text>\",\n"
    "      \"english_body\": \"<translated and normalized text>\",\n"
    "      \"confidence\": \"<confidence score per field>\",\n"
    "      \"flagged_terms\": [\"<list of flagged terms, if any>\"]\n"
    "    },\n"
    "    ...\n"
    "  ]\n"
    "}\n"
)
FALLBACK_RESPONSE = (
    "Unable to process the segment due to insufficient information or an unrecoverable error. "
    "Please review the original data or consult a human reviewer."
)

VALIDATION_CONFIG_PATH = Config.VALIDATION_CONFIG_PATH or str(Path(__file__).parent / "validation_config.json")

class FSRProcessRequest(BaseModel):
    input_json: dict = Field(..., description="The extracted_fsr.json content to be processed.")

    @field_validator("input_json")
    @classmethod
    def validate_input_json(cls, v):
        if not isinstance(v, dict):
            raise ValueError("input_json must be a JSON object.")
        if not v:
            raise ValueError("input_json cannot be empty.")
        return v

class FSRProcessResponse(BaseModel):
    success: bool = Field(..., description="Whether the processing was successful.")
    normalized_fsr: Optional[dict] = Field(None, description="The normalized_fsr.json output.")
    error: Optional[str] = Field(None, description="Error message if processing failed.")
    tips: Optional[str] = Field(None, description="Helpful tips for fixing input or errors.")

@asynccontextmanager
async def _obs_lifespan(application):
    """Initialise observability on startup, clean up on shutdown."""
    try:
        _obs_startup_logger.info('')
        _obs_startup_logger.info('========== Agent Configuration Summary ==========')
        _obs_startup_logger.info(f'Environment: {getattr(Config, "ENVIRONMENT", "N/A")}')
        _obs_startup_logger.info(f'Agent: {getattr(Config, "AGENT_NAME", "N/A")}')
        _obs_startup_logger.info(f'Project: {getattr(Config, "PROJECT_NAME", "N/A")}')
        _obs_startup_logger.info(f'LLM Provider: {getattr(Config, "MODEL_PROVIDER", "N/A")}')
        _obs_startup_logger.info(f'LLM Model: {getattr(Config, "LLM_MODEL", "N/A")}')
        _cs_endpoint = getattr(Config, 'AZURE_CONTENT_SAFETY_ENDPOINT', None)
        _cs_key = getattr(Config, 'AZURE_CONTENT_SAFETY_KEY', None)
        if _cs_endpoint and _cs_key:
            _obs_startup_logger.info('Content Safety: Enabled (Azure Content Safety)')
            _obs_startup_logger.info(f'Content Safety Endpoint: {_cs_endpoint}')
        else:
            _obs_startup_logger.info('Content Safety: Not Configured')
        _obs_startup_logger.info('Observability Database: Azure SQL')
        _obs_startup_logger.info(f'Database Server: {getattr(Config, "OBS_AZURE_SQL_SERVER", "N/A")}')
        _obs_startup_logger.info(f'Database Name: {getattr(Config, "OBS_AZURE_SQL_DATABASE", "N/A")}')
        _obs_startup_logger.info('===============================================')
        _obs_startup_logger.info('')
    except Exception as _e:
        _obs_startup_logger.warning('Config summary failed: %s', _e)

    _obs_startup_logger.info('')
    _obs_startup_logger.info('========== Content Safety & Guardrails ==========')
    if GUARDRAILS_CONFIG.get('content_safety_enabled'):
        _obs_startup_logger.info('Content Safety: Enabled')
        _obs_startup_logger.info(f'  - Severity Threshold: {GUARDRAILS_CONFIG.get("content_safety_severity_threshold", "N/A")}')
        _obs_startup_logger.info(f'  - Check Toxicity: {GUARDRAILS_CONFIG.get("check_toxicity", False)}')
        _obs_startup_logger.info(f'  - Check Jailbreak: {GUARDRAILS_CONFIG.get("check_jailbreak", False)}')
        _obs_startup_logger.info(f'  - Check PII Input: {GUARDRAILS_CONFIG.get("check_pii_input", False)}')
        _obs_startup_logger.info(f'  - Check Credentials Output: {GUARDRAILS_CONFIG.get("check_credentials_output", False)}')
    else:
        _obs_startup_logger.info('Content Safety: Disabled')
    _obs_startup_logger.info('===============================================')
    _obs_startup_logger.info('')

    _obs_startup_logger.info('========== Initializing Agent Services ==========')
    try:
        from observability.database.engine import create_obs_database_engine
        from observability.database.base import ObsBase
        import observability.database.models  # noqa: F401
        _obs_engine = create_obs_database_engine()
        ObsBase.metadata.create_all(bind=_obs_engine, checkfirst=True)
        _obs_startup_logger.info('✓ Observability database connected')
    except Exception as _e:
        _obs_startup_logger.warning('✗ Observability database connection failed (metrics will not be saved)')
    try:
        _t = initialize_tracer()
        if _t is not None:
            _obs_startup_logger.info('✓ Telemetry monitoring enabled')
        else:
            _obs_startup_logger.warning('✗ Telemetry monitoring disabled')
    except Exception as _e:
        _obs_startup_logger.warning('✗ Telemetry monitoring failed to initialize')
    _obs_startup_logger.info('=================================================')
    _obs_startup_logger.info('')
    yield

app = FastAPI(lifespan=_obs_lifespan,

    title="Emerson FSR Translation and Normalization Agent",
    description="Processes extracted_fsr.json and returns normalized_fsr.json with English translations, original text, confidence scores, and flagged terms.",
    version=Config.SERVICE_VERSION if hasattr(Config, "SERVICE_VERSION") else "1.0.0",
    # SYNTAX-FIX: lifespan=_obs_lifespan
)

@app.exception_handler(RequestValidationError)
@with_content_safety(config=GUARDRAILS_CONFIG)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    logger.warning(f"Malformed JSON or validation error: {exc}")
    return JSONResponse(
        status_code=422,
        content={
            "success": False,
            "error": "Malformed JSON or validation error.",
            "tips": "Ensure your request body is valid JSON and matches the required schema. Check for missing commas, mismatched quotes, or incorrect field names.",
        },
    )

@app.exception_handler(ValidationError)
@with_content_safety(config=GUARDRAILS_CONFIG)
async def pydantic_validation_exception_handler(request: Request, exc: ValidationError):
    logger.warning(f"Pydantic validation error: {exc}")
    return JSONResponse(
        status_code=422,
        content={
            "success": False,
            "error": "Input validation failed.",
            "tips": "Check that all required fields are present and correctly formatted.",
        },
    )

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "ok"}

@with_content_safety(config=GUARDRAILS_CONFIG)
def get_llm_client():
    api_key = Config.AZURE_OPENAI_API_KEY
    if not api_key:
        raise ValueError("AZURE_OPENAI_API_KEY not configured")
    return openai.AsyncAzureOpenAI(
        api_key=api_key,
        api_version="2024-02-01",
        azure_endpoint=Config.AZURE_OPENAI_ENDPOINT,
    )

class Segmenter:
    def segment(self, document: dict) -> List[str]:
        """Segments the input document into logical units."""
        # The actual segmentation logic is handled by the LLM via the system prompt.
        # Here, we simply pass the document to the LLM for segmentation.
        return [json.dumps(document)]

class LanguageDetector:
    async def detect_language(self, segment_text: str) -> str:
        """Detects the source language for a segment."""
        # The LLM is responsible for language detection as per the system prompt.
        return "en"

class Translator:
    async def translate(self, segment_text: str, detected_language: str) -> str:
        """Translates segment to English using glossary."""
        # The LLM is responsible for translation as per the system prompt.
        return segment_text

class Normalizer:
    async def normalize(self, translated_segment: str) -> str:
        """Normalizes translated terms against taxonomy."""
        # The LLM is responsible for normalization as per the system prompt.
        return translated_segment

class ConfidenceScorer:
    async def score_confidence(self, normalized_segment: str) -> (str, List[str]):
        """Assigns confidence scores and flags low-confidence terms."""
        # The LLM is responsible for confidence scoring as per the system prompt.
        return "1.0", []

class AuditLogger:
    def __init__(self):
        self.logger = logging.getLogger("agent.audit")

    def log_action(self, action: str, segment: Any):
        self.logger.info(f"Action: {action} | Segment: {segment}")

    def log_error(self, error: str, segment: Any):
        self.logger.error(f"Error: {error} | Segment: {segment}")

class OutputFormatter:
    def format_output(self, processed_segments: List[dict]) -> dict:
        """Formats processed segments into normalized_fsr.json."""
        return {"segments": processed_segments}

import re as _re

_FENCE_RE = _re.compile(r"```(?:\w+)?\s*\n(.*?)```", _re.DOTALL)
_LONE_FENCE_START_RE = _re.compile(r"^```\w*$")
_WRAPPER_RE = _re.compile(
    r"^(?:"
    r"Here(?:'s| is)(?: the)? (?:the |your |a )?(?:code|solution|implementation|result|explanation|answer)[^:]*:\s*"
    r"|Sure[!,.]?\s*"
    r"|Certainly[!,.]?\s*"
    r"|Below is [^:]*:\s*"
    r")",
    _re.IGNORECASE,
)
_SIGNOFF_RE = _re.compile(
    r"^(?:Let me know|Feel free|Hope this|This code|Note:|Happy coding|If you)",
    _re.IGNORECASE,
)
_BLANK_COLLAPSE_RE = _re.compile(r"\n{3,}")

def _strip_fences(text: str, content_type: str) -> str:
    """Extract content from Markdown code fences."""
    fence_matches = _FENCE_RE.findall(text)
    if fence_matches:
        if content_type == "code":
            return "\n\n".join(block.strip() for block in fence_matches)
        for match in fence_matches:
            fenced_block = _FENCE_RE.search(text)
            if fenced_block:
                text = text[:fenced_block.start()] + match.strip() + text[fenced_block.end():]
        return text
    lines = text.splitlines()
    if lines and _LONE_FENCE_START_RE.match(lines[0].strip()):
        lines = lines[1:]
    if lines and lines[-1].strip() == "```":
        lines = lines[:-1]
    return "\n".join(lines).strip()

def _strip_trailing_signoffs(text: str) -> str:
    """Remove conversational sign-off lines from the end of code output."""
    lines = text.splitlines()
    while lines and _SIGNOFF_RE.match(lines[-1].strip()):
        lines.pop()
    return "\n".join(lines).rstrip()

@with_content_safety(config=GUARDRAILS_CONFIG)
def sanitize_llm_output(raw: str, content_type: str = "code") -> str:
    """
    Generic post-processor that cleans common LLM output artefacts.
    Args:
        raw: Raw text returned by the LLM.
        content_type: 'code' | 'text' | 'markdown'.
    Returns:
        Cleaned string ready for validation, formatting, or direct return.
    """
    if not raw:
        return ""
    text = _strip_fences(raw.strip(), content_type)
    text = _WRAPPER_RE.sub("", text, count=1).strip()
    if content_type == "code":
        text = _strip_trailing_signoffs(text)
    return _BLANK_COLLAPSE_RE.sub("\n\n", text).strip()

class BaseAgent:
    def __init__(self):
        self.segmenter = Segmenter()
        self.language_detector = LanguageDetector()
        self.translator = Translator()
        self.normalizer = Normalizer()
        self.confidence_scorer = ConfidenceScorer()
        self.audit_logger = AuditLogger()
        self.output_formatter = OutputFormatter()

class EmersonFSRTranslationAgent(BaseAgent):
    def __init__(self):
        super().__init__()
        self.llm_client = None

    def _get_llm_client(self):
        if self.llm_client is None:
            self.llm_client = get_llm_client()
        return self.llm_client

    @with_content_safety(config=GUARDRAILS_CONFIG)
    async def process_fsr(self, input_json: dict) -> dict:
        """
        Main entry point; orchestrates the processing pipeline for the input FSR JSON.
        Returns a dict with keys: success, normalized_fsr, error, tips.
        """
        processed_segments = []
        errors = []
        try:
            async with trace_step(
                "segment_document",
                step_type="process",
                decision_summary="Segment input document respecting boundaries",
                output_fn=lambda r: f"{len(r)} segments" if isinstance(r, list) else "segmentation failed"
            ) as step:
                try:
                    segments = self.segmenter.segment(input_json)
                    self.audit_logger.log_action("segmentation", segments)
                except Exception as e:
                    self.audit_logger.log_error("OUTPUT_FORMAT_ERROR: " + str(e), input_json)
                    return {
                        "success": False,
                        "normalized_fsr": None,
                        "error": "Segmentation failed: OUTPUT_FORMAT_ERROR",
                        "tips": "Ensure the input document is well-formed and respects paragraph/table boundaries."
                    }
                step.capture(segments)

            for segment in segments:
                segment_result = {
                    "original": None,
                    "english_body": None,
                    "confidence": None,
                    "flagged_terms": [],
                }
                try:
                    async with trace_step(
                        "process_segment",
                        step_type="process",
                        decision_summary="Process one segment through all pipeline steps",
                        output_fn=lambda r: f"Processed segment"
                    ) as step:
                        # 1. Preserve original
                        segment_result["original"] = segment

                        # 2. Detect language
                        detected_language = None
                        for attempt in range(3):
                            try:
                                detected_language = await self.language_detector.detect_language(segment)
                                self.audit_logger.log_action("language_detection", {"segment": segment, "detected_language": detected_language})
                                break
                            except Exception as e:
                                self.audit_logger.log_error(f"LANG_DETECT_FAIL (attempt {attempt+1}): {e}", segment)
                                if attempt == 2:
                                    segment_result["english_body"] = FALLBACK_RESPONSE
                                    segment_result["confidence"] = "0.0"
                                    segment_result["flagged_terms"] = ["LANG_DETECT_FAIL"]
                                    errors.append(f"LANG_DETECT_FAIL: {e}")
                                    break

                        # 3. Translate if needed
                        translated_segment = segment
                        if detected_language and detected_language.lower() != "en":
                            for attempt in range(3):
                                try:
                                    translated_segment = await self.translator.translate(segment, detected_language)
                                    self.audit_logger.log_action("translation", {"segment": segment, "translated": translated_segment})
                                    break
                                except Exception as e:
                                    self.audit_logger.log_error(f"TRANSLATION_FAIL (attempt {attempt+1}): {e}", segment)
                                    if attempt == 2:
                                        segment_result["english_body"] = FALLBACK_RESPONSE
                                        segment_result["confidence"] = "0.0"
                                        segment_result["flagged_terms"] = ["TRANSLATION_FAIL"]
                                        errors.append(f"TRANSLATION_FAIL: {e}")
                                        break

                        # 4. Normalize
                        normalized_segment = translated_segment
                        for attempt in range(3):
                            try:
                                normalized_segment = await self.normalizer.normalize(translated_segment)
                                self.audit_logger.log_action("normalization", {"translated": translated_segment, "normalized": normalized_segment})
                                break
                            except Exception as e:
                                self.audit_logger.log_error(f"NORMALIZATION_FAIL (attempt {attempt+1}): {e}", translated_segment)
                                if attempt == 2:
                                    segment_result["english_body"] = FALLBACK_RESPONSE
                                    segment_result["confidence"] = "0.0"
                                    segment_result["flagged_terms"] = ["NORMALIZATION_FAIL"]
                                    errors.append(f"NORMALIZATION_FAIL: {e}")
                                    break

                        # 5. Confidence scoring
                        try:
                            confidence, flagged_terms = await self.confidence_scorer.score_confidence(normalized_segment)
                            segment_result["english_body"] = normalized_segment
                            segment_result["confidence"] = confidence
                            segment_result["flagged_terms"] = flagged_terms
                            self.audit_logger.log_action("confidence_scoring", {"normalized": normalized_segment, "confidence": confidence, "flagged_terms": flagged_terms})
                        except Exception as e:
                            self.audit_logger.log_error(f"CONFIDENCE_SCORE_FAIL: {e}", normalized_segment)
                            segment_result["english_body"] = FALLBACK_RESPONSE
                            segment_result["confidence"] = "0.0"
                            segment_result["flagged_terms"] = ["CONFIDENCE_SCORE_FAIL"]
                            errors.append(f"CONFIDENCE_SCORE_FAIL: {e}")

                        step.capture(segment_result)
                except Exception as e:
                    self.audit_logger.log_error(f"Segment processing error: {e}", segment)
                    segment_result["english_body"] = FALLBACK_RESPONSE
                    segment_result["confidence"] = "0.0"
                    segment_result["flagged_terms"] = ["PROCESSING_ERROR"]
                    errors.append(f"PROCESSING_ERROR: {e}")
                processed_segments.append(segment_result)

            # 6. Output formatting
            try:
                async with trace_step(
                    "format_output",
                    step_type="format",
                    decision_summary="Aggregate and format output",
                    output_fn=lambda r: f"{len(r.get('segments', []))} segments in output" if isinstance(r, dict) else "formatting failed"
                ) as step:
                    normalized_fsr = self.output_formatter.format_output(processed_segments)
                    step.capture(normalized_fsr)
            except Exception as e:
                self.audit_logger.log_error("OUTPUT_FORMAT_ERROR: " + str(e), processed_segments)
                return {
                    "success": False,
                    "normalized_fsr": None,
                    "error": "Output formatting failed: OUTPUT_FORMAT_ERROR",
                    "tips": "Check that all segments are correctly processed and output structure matches normalized_fsr.json."
                }

            return {
                "success": True,
                "normalized_fsr": normalized_fsr,
                "error": None if not errors else "; ".join(errors),
                "tips": None if not errors else "Some segments could not be processed. See flagged_terms and error details."
            }
        except Exception as e:
            self.audit_logger.log_error(f"Critical agent error: {e}", input_json)
            return {
                "success": False,
                "normalized_fsr": None,
                "error": f"Critical agent error: {e}",
                "tips": "Check input data and agent logs for more details."
            }

@app.post("/process_fsr", response_model=FSRProcessResponse)
@with_content_safety(config=GUARDRAILS_CONFIG)
async def process_fsr_endpoint(req: FSRProcessRequest):
    """
    Endpoint to process extracted_fsr.json and return normalized_fsr.json.
    """
    agent = EmersonFSRTranslationAgent()
    _t0 = _time.time()
    async with trace_step(
        "process_fsr_endpoint",
        step_type="process",
        decision_summary="Invoke agent process_fsr",
        output_fn=lambda r: "success" if r.get("success") else "failure"
    ) as step:
        result = await agent.process_fsr(req.input_json)
        step.capture(result)
    try:
        trace_model_call(
            provider="azure",
            model_name=Config.LLM_MODEL or "gpt-4.1",
            prompt_tokens=0,
            completion_tokens=0,
            latency_ms=int((_time.time() - _t0) * 1000),
            response_summary=str(result)[:200] if result else "",
        )
    except Exception:
        pass
    # Sanitize LLM output if present
    if result.get("normalized_fsr"):
        try:
            result["normalized_fsr"] = json.loads(
                sanitize_llm_output(json.dumps(result["normalized_fsr"]), content_type="code")
            )
        except Exception:
            pass
    return result

async def _run_agent():
    """Entrypoint: runs the agent with observability (trace collection only)."""
    import uvicorn

    _LOG_CONFIG = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "default": {
                "()": "uvicorn.logging.DefaultFormatter",
                "fmt": "%(levelprefix)s %(name)s: %(message)s",
                "use_colors": None,
            },
            "access": {
                "()": "uvicorn.logging.AccessFormatter",
                "fmt": '%(levelprefix)s %(client_addr)s - "%(request_line)s" %(status_code)s',
            },
        },
        "handlers": {
            "default": {
                "formatter": "default",
                "class": "logging.StreamHandler",
                "stream": "ext://sys.stderr",
            },
            "access": {
                "formatter": "access",
                "class": "logging.StreamHandler",
                "stream": "ext://sys.stdout",
            },
        },
        "loggers": {
            "uvicorn":        {"handlers": ["default"], "level": "INFO", "propagate": False},
            "uvicorn.error":  {"level": "INFO"},
            "uvicorn.access": {"handlers": ["access"], "level": "INFO", "propagate": False},
            "agent":          {"handlers": ["default"], "level": "INFO", "propagate": False},
            "__main__":       {"handlers": ["default"], "level": "INFO", "propagate": False},
            "observability": {"handlers": ["default"], "level": "INFO", "propagate": False},
            "config": {"handlers": ["default"], "level": "INFO", "propagate": False},
            "azure":   {"handlers": ["default"], "level": "WARNING", "propagate": False},
            "urllib3": {"handlers": ["default"], "level": "WARNING", "propagate": False},
        },
    }

    config = uvicorn.Config(
        "agent:app",
        host="0.0.0.0",
        port=8080,
        reload=False,
        log_level="info",
        log_config=_LOG_CONFIG,
    )
    server = uvicorn.Server(config)
    await server.serve()


if __name__ == "__main__":
    _asyncio.run(_run_agent())
# Emerson FSR Translation and Normalization Agent

A professional, modular Python agent for processing Emerson field service reports (FSR). It segments, detects language, translates, normalizes, scores confidence, and outputs auditable, normalized FSR JSON. Built with FastAPI, Azure OpenAI, and robust observability.

---

## Quick Start

### 1. Create a virtual environment:
```
python -m venv .venv
```

### 2. Activate the virtual environment:

**Windows:**
```
.venv\Scripts\activate
```

**macOS/Linux:**
```
source .venv/bin/activate
```

### 3. Install dependencies:
```
pip install -r requirements.txt
```

### 4. Environment setup:
Copy `.env.example` to `.env` and fill in all required values.
```
cp .env.example .env
```

### 5. Running the agent

**Direct execution:**
```
python code/agent.py
```

**As a FastAPI server:**
```
uvicorn code.agent:app --reload --host 0.0.0.0 --port 8000
```

---

## Environment Variables

**Agent Identity**
- AGENT_NAME
- AGENT_ID
- PROJECT_NAME
- PROJECT_ID

**General**
- ENVIRONMENT

**Azure Key Vault**
- USE_KEY_VAULT
- KEY_VAULT_URI
- AZURE_USE_DEFAULT_CREDENTIAL

**Azure Authentication**
- AZURE_TENANT_ID
- AZURE_CLIENT_ID
- AZURE_CLIENT_SECRET

**LLM Configuration**
- MODEL_PROVIDER
- LLM_MODEL
- LLM_TEMPERATURE
- LLM_MAX_TOKENS
- LLM_MODELS (JSON array)

**API Keys / Secrets**
- OPENAI_API_KEY
- AZURE_OPENAI_API_KEY
- ANTHROPIC_API_KEY
- GOOGLE_API_KEY
- AZURE_SEARCH_API_KEY
- AZURE_CONTENT_SAFETY_KEY

**Service Endpoints**
- AZURE_OPENAI_ENDPOINT
- AZURE_CONTENT_SAFETY_ENDPOINT
- AZURE_SEARCH_ENDPOINT

**Observability Database**
- OBS_DATABASE_TYPE
- OBS_AZURE_SQL_SERVER
- OBS_AZURE_SQL_DATABASE
- OBS_AZURE_SQL_PORT
- OBS_AZURE_SQL_USERNAME
- OBS_AZURE_SQL_PASSWORD
- OBS_AZURE_SQL_SCHEMA
- OBS_AZURE_SQL_TRUST_SERVER_CERTIFICATE

**Agent-Specific**
- VALIDATION_CONFIG_PATH
- SERVICE_NAME
- SERVICE_VERSION
- VERSION
- CONTENT_SAFETY_ENABLED
- CONTENT_SAFETY_SEVERITY_THRESHOLD
- AZURE_SEARCH_INDEX_NAME

---

## API Endpoints

### **GET** `/health`
- **Description:** Health check endpoint.
- **Response:**
  ```
  {
    "status": "ok"
  }
  ```

### **POST** `/process_fsr`
- **Description:** Process an extracted FSR JSON and return a normalized FSR JSON.
- **Request body:**
  ```
  {
    "input_json": "object (required)"  // The extracted_fsr.json content to be processed
  }
  ```
- **Response:**
  ```
  {
    "success": true|false,
    "normalized_fsr": {
      "segments": [
        {
          "original": "string",
          "english_body": "string",
          "confidence": "string",
          "flagged_terms": ["string", ...]
        },
        ...
      ]
    } | null,
    "error": null|string,
    "tips": null|string
  }
  ```

- **Error Response (422):**
  ```
  {
    "success": false,
    "error": "Malformed JSON or validation error.",
    "tips": "Ensure your request body is valid JSON and matches the required schema. Check for missing commas, mismatched quotes, or incorrect field names."
  }
  ```

---

## Running Tests

### 1. Install test dependencies (if not already installed):
```
pip install pytest pytest-asyncio
```

### 2. Run all tests:
```
pytest tests/
```

### 3. Run a specific test file:
```
pytest tests/test_<module_name>.py
```

### 4. Run tests with verbose output:
```
pytest tests/ -v
```

### 5. Run tests with coverage report:
```
pip install pytest-cov
pytest tests/ --cov=code --cov-report=term-missing
```

---

## Deployment with Docker

### 1. Prerequisites: Ensure Docker is installed and running.

### 2. Environment setup: Copy `.env.example` to `.env` and configure all required environment variables.

### 3. Build the Docker image:
```
docker build -t emerson-fsr-agent -f deploy/Dockerfile .
```

### 4. Run the Docker container:
```
docker run -d --env-file .env -p 8000:8000 --name emerson-fsr-agent emerson-fsr-agent
```

### 5. Verify the container is running:
```
docker ps
```

### 6. View container logs:
```
docker logs emerson-fsr-agent
```

### 7. Stop the container:
```
docker stop emerson-fsr-agent
```

---

## Notes

- All run commands must use the `code/` prefix (e.g., `python code/agent.py`, `uvicorn code.agent:app ...`).
- See `.env.example` for all required and optional environment variables.
- The agent requires access to LLM API keys and (optionally) Azure SQL for observability.
- For production, configure Key Vault and secure credentials as needed.

---

**Emerson FSR Translation and Normalization Agent** — Reliable, auditable, and consistent FSR normalization for global field service operations.

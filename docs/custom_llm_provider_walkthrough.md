# Custom HTTP LLM Provider Integration Walkthrough

## 1. Overview of Current Architecture
The SQL Assistant currently supports two main paths for LLM provider integration, routed through a central factory (`src/llm_factory.py`):
- **Ollama (`provider="ollama"`)**: Local models managed via `OllamaManager`.
- **Enterprise / OpenAI (`provider="enterprise"`)**: Remote models accessed via standard `crewai.LLM` leveraging LiteLLM and environment variables like `ENTERPRISE_LLM_API_KEY` and `ENTERPRISE_LLM_ENDPOINT`.

These providers are instantiated centrally and mapped into `Crew` agents (via `SQLAgent` in `src/sql_agent.py`), which abstracts the actual LLM engine away from the agent logic. The Streamlit UI (`app.py`) dynamically queries the factory to display available models relying on backend capabilities.

## 2. Proposed Design: Custom HTTP Provider
We are adding a **third provider type** (`custom_http`), specifically to call a newly introduced inference API endpoint:
`https://xxxx.in/completions`

### Environment Variables (.env)
These values will be specified exclusively via environment settings to ensure no secrets are stored in code and to maintain flexible configuration across deployments:
- `CUSTOM_LLM_ENABLED`: Toggle to selectively enable the new provider (default: `false`).
- `CUSTOM_LLM_API_BASE`: Endpoint base URL (default: `https://xxxx.in/completions`).
- `CUSTOM_LLM_API_KEY`: The API key (passed within the `Authorization: Bearer <API_KEY>` header).
- `CUSTOM_LLM_MODEL_DEFAULT`: The model name to pass within the JSON body (e.g., `chat-model`).

### Security Guarantee
**No secrets will be hard-coded.** All access tokens and API keys will be fetched via environment configurations mirroring the `ENTERPRISE_LLM` structures, and will not be routinely logged in debug outputs.

## 3. Changes to the LLM Factory (`src/llm_factory.py`)
To ensure that branching logic is isolated entirely to the instantiation layer (leaving agent tools and UI untouched), we will make the following additions:

1. **Extend `get_combined_model_list`**:
   - Check if `os.getenv("CUSTOM_LLM_ENABLED", "false").lower() == "true"`.
   - If enabled, inject a new entry into the unified dropdown options:
     ```python
     {
         "provider": "custom_http",
         "model": os.getenv("CUSTOM_LLM_MODEL_DEFAULT", "chat-model"),
         "label": f"Custom API \u2013 {model}"
     }
     ```

2. **Extend `create_llm`**:
   - Add a routing condition: `if provider == "custom_http": return _create_custom_http_llm(...)`.

3. **Implement the HTTP Adapter (`_create_custom_http_llm`)**:
   - The CrewAI/Langchain models expect specific interfaces (`.invoke`, `.call`, etc.).
   - We will introduce a lightweight wrapper class extending the standard ChatModel base. This class will internally handle the specific requests format necessary:
     - **Request Mapping**: Extract the `SystemMessage` and `HumanMessage` prompts, and pack them into the precise `{ "model": "...", "messages": [...], "temperature": 0.0 }` JSON array.
     - **Execution**: Issue a `POST` request to the target gateway endpoint using standard Python `requests` with the `{ "Authorization": "Bearer <API_KEY>" }` header.
     - **Response Mapping**: Extract the text strings out of the nested JSON HTTP response and construct a standardized Langchain AIMessage for transparent consumption by the rest of the application.

## 4. Updates to Streamlit UI
Because the "Model Configuration" sidebar logic in `app.py` already programmatically renders entries yielded by `get_combined_model_list()`, **the UI requires virtually no changes** to display the Custom HTTP model.
Changing the model selection drop-down locally will implicitly update the `st.session_state.llm_provider` payload to `"custom_http"` safely, which is subsequently passed to `SQLAgent.update_llm(provider, model)`.

## 5. Backward Compatibility Guarantee
**No existing behavior will be permanently modified.**
- Existing implementations for `ollama` generation strictly persist.
- Existing logic for `enterprise` generation correctly bypasses the new conditions.
- Error behaviors (e.g., API timeouts) will be handled congruently with existing implementations.
- If `CUSTOM_LLM_ENABLED=false` or is omitted, the application operates perfectly inline with its current production state.

## 6. Implementation Checklist
The following steps outline the required codebase changes to be implemented once this document is reviewed and approved:
- [ ] Add parsing for `CUSTOM_LLM_*` environment variables in the factory.
- [ ] Add `CustomHTTPChatLLM` class matching the Langchain interface but utilizing HTTP requests internally.
- [ ] Add `_create_custom_http_llm` logic within `src/llm_factory.py`.
- [ ] Wire the list population in `get_combined_model_list` for the Custom provider.
- [ ] Update `env.example` mapping to indicate the inclusion of the new parameters.

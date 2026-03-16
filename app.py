# __import__("pysqlite3")
# import sys
# sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")

import streamlit as st
import pandas as pd
import time
import plotly.express as px
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Set dummy OpenAI API key to bypass CrewAI/LangChain strict validation
# This is required even when using local Ollama models via ChatOpenAI
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY", "NA")
os.environ["OPENAI_API_BASE"] = os.getenv("OPENAI_API_BASE", "http://localhost:11434/v1")
os.environ["OPENAI_BASE_URL"] = os.getenv("OPENAI_BASE_URL", "http://localhost:11434/v1")
os.environ["OPENAI_MODEL_NAME"] = os.getenv("OPENAI_MODEL_NAME", "sqlcoder:7b")  # Default model, updated dynamically

from src.sql_agent import SQLAgent
from src.sql_agent import get_sql_agent_mode, get_effective_mode
from src.utils import (
    load_config, save_config, format_time_taken,
    load_settings_from_db, save_settings_to_db,
    init_settings_table, save_feedback_to_db,
    normalize_user_id, validate_user_input_v2
)

# Constants
CONFIG_PATH = "config/database_config.json"

# Page configuration
st.set_page_config(
    page_title="SQL Assistant",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .stChatMessage {
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    .stChatMessage[data-testid="stChatMessageUser"] {
        background-color: #f0f2f6;
    }
    .stChatMessage[data-testid="stChatMessageAssistant"] {
        background-color: #ffffff;
        border: 1px solid #e0e0e0;
    }
    .response-time {
        font-size: 0.8rem;
        color: #666;
        margin-top: 0.5rem;
        font-style: italic;
    }
    .user-id-badge {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 600;
        display: inline-block;
    }
    .modification-banner {
        background-color: #fff3cd;
        border-left: 4px solid #ffc107;
        padding: 6px 12px;
        margin-bottom: 8px;
        border-radius: 4px;
        font-size: 0.85rem;
    }
</style>
""", unsafe_allow_html=True)


def initialize_session_state():
    """Initialize session state variables."""
    # Per-user message dict: {user_id: [msg, ...]}
    if 'messages' not in st.session_state:
        st.session_state.messages = {}
    if 'sql_agent' not in st.session_state:
        st.session_state.sql_agent = None
    if 'db_connected' not in st.session_state:
        st.session_state.db_connected = False
    if 'ollama_connected' not in st.session_state:
        st.session_state.ollama_connected = False

    # User authentication state
    if 'user_authenticated' not in st.session_state:
        st.session_state.user_authenticated = False
    if 'user_id' not in st.session_state:
        st.session_state.user_id = None          # String format: "emp:123" or "mob:9876543210"
    if 'session_id' not in st.session_state:
        st.session_state.session_id = None
    if 'user_info' not in st.session_state:
        st.session_state.user_info = {}

    # Load local config
    if 'config' not in st.session_state:
        st.session_state.config = load_config(CONFIG_PATH)

    # LLM provider/model selection (UI state — overrides env defaults)
    if 'llm_provider' not in st.session_state:
        st.session_state.llm_provider = os.getenv("LLM_PROVIDER_DEFAULT", "ollama")
    if 'llm_model' not in st.session_state:
        st.session_state.llm_model = os.getenv("LLM_MODEL_DEFAULT", "")
    # Agent mode selection (UI state — overrides env default)
    if 'agent_mode' not in st.session_state:
        default_mode = (
            os.getenv("SQL_AGENT_MODE") or
            os.getenv("SQL_AGENT_MODE_DEFAULT", "multi")
        ).strip().lower()
        st.session_state.agent_mode = "single" if default_mode == "single" else "multi"

    # Try to connect to DB on startup (without user_id)
    if not st.session_state.db_connected and st.session_state.sql_agent is None:
        try:
            agent = SQLAgent(CONFIG_PATH)
            if agent.db_manager.connect():
                st.session_state.sql_agent = agent
                st.session_state.db_connected = True

                from src.utils import init_user_tables
                init_user_tables(agent.db_manager)

                if not agent.load_configured_schema():
                    ddls = agent.db_manager.get_all_ddls()
                    agent.vector_store.store_ddls(ddls)

                init_settings_table(agent.db_manager)

                db_settings = load_settings_from_db(agent.db_manager)
                if db_settings:
                    if 'ollama' in db_settings:
                        st.session_state.config['ollama'] = db_settings['ollama']
                    if 'settings' in db_settings:
                        st.session_state.config['settings'] = db_settings['settings']
                    st.toast("Settings loaded from Database")
        except Exception as e:
            st.error(f"Initialization Error: {str(e)}")


def save_current_settings():
    """Save current sidebar settings."""
    new_config = st.session_state.config.copy()
    new_config['ollama']['base_url'] = st.session_state.ollama_url
    new_config['ollama']['model'] = st.session_state.ollama_model
    if 'settings' not in new_config:
        new_config['settings'] = {}
    new_config['settings']['temperature'] = st.session_state.temperature
    new_config['settings']['max_tokens'] = st.session_state.max_tokens

    if save_config(CONFIG_PATH, new_config):
        st.session_state.config = new_config
        st.toast("Local config saved")

    if st.session_state.db_connected and st.session_state.sql_agent:
        db_settings = {"ollama": new_config['ollama'], "settings": new_config['settings']}
        if save_settings_to_db(st.session_state.sql_agent.db_manager, db_settings):
            st.toast("Preferences saved to Database")
            st.session_state.sql_agent.update_model(
                model_name=new_config['ollama']['model'],
                base_url=new_config['ollama']['base_url']
            )
        else:
            st.error("Failed to save to Database")
    else:
        st.warning("Connect to Database to save preferences remotely")


@st.dialog("Agent Performance Diagnostics", width="large")
def show_diagnostics_dialog(metrics):
    import pandas as pd
    chroma = metrics.get('chroma', {})
    agents = metrics.get('agent_pipeline', [])
    
    st.markdown("##### 🔬 Semantic Retrieval Logs")
    for r_log in chroma.get('recent_logs', [])[:5]:
        try:
            sc = r_log.get('relevance_scores', {})
            if isinstance(sc, str):
                import json
                sc = json.loads(sc)
            scores_str = ", ".join([f"{k}({v})" for k, v in sc.items()][:2])
            mins = r_log.get("chroma_query_time_ms", 0) / 60000.0
            st.caption(f'"{r_log["user_prompt"]}" → {scores_str} [{mins:.4f} mins]')
        except:
            mins = r_log.get("chroma_query_time_ms", 0) / 60000.0
            st.caption(f'"{r_log.get("user_prompt", "")}" → {r_log.get("matched_tables")} [{mins:.4f} mins]')
    
    st.divider()
    st.markdown("##### 🔍 Agent Tracing")
    if agents:
        agents_df = pd.DataFrame(agents)
        if 'avg_time' in agents_df.columns:
            agents_df['avg_time_mins'] = (agents_df['avg_time'] / 60000.0).round(4).astype(str) + ' mins'
        st.dataframe(agents_df, use_container_width=True)
    
    st.markdown("##### ⚡ Raw Logs")
    raw_logs = metrics.get('raw_logs', [])
    if raw_logs:
        raw_df = pd.DataFrame(raw_logs)
        if 'time_taken_ms' in raw_df.columns:
            raw_df['time_taken_mins'] = (raw_df['time_taken_ms'] / 60000.0).round(4).astype(str) + ' mins'
        st.dataframe(raw_df, use_container_width=True)

def sidebar_settings():
    """Render the settings sidebar."""
    with st.sidebar:
        st.title("⚙ Settings")

        # ── User Identification Panel ──────────────────────────────────────
        with st.expander("👤 User Identity", expanded=not st.session_state.user_authenticated):
            if not st.session_state.user_authenticated:
                st.info("Enter **Employee ID** OR **Mobile Number** to start")

                emp_id = st.text_input(
                    "Employee ID", key="input_emp_id",
                    placeholder="e.g. 12345",
                    help="Will become emp:12345"
                )
                mobile = st.text_input(
                    "Mobile Number", key="input_mobile",
                    placeholder="e.g. 9876543210",
                    help="Will become mob:9876543210"
                )

                if st.button("🚀 Start Session", use_container_width=True):
                    # Use new v2 validator that requires emp_id or mobile
                    is_valid, error_msg, user_id = validate_user_input_v2(emp_id, mobile)

                    if not is_valid:
                        st.error(error_msg)
                    elif not st.session_state.db_connected:
                        st.error("Database not connected. Cannot start session.")
                    else:
                        from src.utils import get_db_user_id
                        db_user_id = get_db_user_id(st.session_state.sql_agent.db_manager, emp_id, mobile)
                        if not db_user_id:
                            st.error("Error creating or fetching user profile from database.")
                            st.stop()
                            
                        with st.spinner(f"Starting session as **{user_id}** (DB ID: {db_user_id})..."):
                            try:
                                import secrets
                                session_id = f"sess_{db_user_id}_{secrets.token_hex(6)}"

                                # Build new SQLAgent with string_user_id mapped to DB integer ID for pg_memory
                                new_agent = SQLAgent(
                                    CONFIG_PATH,
                                    session_id=session_id,
                                    string_user_id=str(db_user_id)
                                )
                                if new_agent.db_manager.connect():
                                    if not new_agent.load_configured_schema():
                                        ddls = new_agent.db_manager.get_all_ddls()
                                        new_agent.vector_store.store_ddls(ddls)

                                    # Initialize PostgresTextMemory tables + session
                                    new_agent.init_pg_memory()

                                    st.session_state.sql_agent = new_agent
                                    st.session_state.user_id = user_id
                                    st.session_state.session_id = session_id
                                    st.session_state.user_authenticated = True
                                    st.session_state.user_info = {
                                        "emp_id": emp_id or "—",
                                        "mobile": mobile or "—"
                                    }

                                    # Initialize per-user message list if not already
                                    if user_id not in st.session_state.messages:
                                        st.session_state.messages[user_id] = []

                                    st.success(f"✅ Session started as **{user_id}**")
                                    st.rerun()
                                else:
                                    st.error("Failed to connect to database.")
                            except Exception as e:
                                st.error(f"Session start error: {str(e)}")
            else:
                # Show current user session info
                user_id = st.session_state.user_id
                st.markdown(
                    f'<div class="user-id-badge">🔑 {user_id}</div>',
                    unsafe_allow_html=True
                )
                st.write(f"**Session:** `{st.session_state.session_id[:20] if st.session_state.session_id else 'N/A'}...`")
                msg_count = len(st.session_state.messages.get(user_id, []))
                st.caption(f"📝 {msg_count} messages in this session")

                if st.button("🔄 New Session", use_container_width=True):
                    st.session_state.user_authenticated = False
                    st.session_state.user_id = None
                    st.session_state.session_id = None
                    st.session_state.sql_agent = None
                    st.session_state.db_connected = False
                    st.rerun()

        st.markdown("---")

        # Agent Diagnostics Dashboard
        if st.session_state.user_authenticated:
            with st.expander("📊 Agent Diagnostics", expanded=False):
                user_id = st.session_state.user_id
                if st.session_state.sql_agent and getattr(st.session_state.sql_agent, 'logger_instance', None):
                    with st.spinner("Loading metrics..."):
                        # Ensure we query metrics matching the db integer user_id
                        agent_user_id = getattr(st.session_state.sql_agent, "string_user_id", user_id)
                        metrics = st.session_state.sql_agent.logger_instance.get_performance_dashboard(agent_user_id)
                        if metrics:
                            chroma = metrics.get('chroma', {})
                            c_mins = chroma.get('avg_time_ms', 0) / 60000.0
                            st.write(f"**Chroma**: {c_mins:.4f} mins")
                            agents = metrics.get('agent_pipeline', [])
                            if agents:
                                pipeline_str = " | ".join([f"{a['agent'][:3].upper()}({a['avg_time']/60000.0:.2f} mins)" for a in agents])
                                st.write(f"**Agents**: {pipeline_str}")
                            st.write(f"**SQL Match**: {metrics.get('overall_sql_success', 0)}%")
                            
                            if st.button("🔍 View Detailed Logs", use_container_width=True):
                                show_diagnostics_dialog(metrics)
                        else:
                            st.info("No metrics yet.")
                            
        st.markdown("---")

        # Connection Status
        if st.session_state.db_connected:
            st.success("✅ Database Connected")
        else:
            st.error("❌ Database Disconnected")
            st.info("Check config/database_config.json")
            if st.button("Retry Connection"):
                st.rerun()

        # Model Settings
        with st.expander("🤖 Model Configuration", expanded=True):
            ollama_config = st.session_state.config.get('ollama', {})
            ui_settings = st.session_state.config.get('settings', {})

            st.text_input("Base URL", value=ollama_config.get('base_url', 'http://localhost:11434'), key='ollama_url')

            # ── Unified model list (Ollama + Enterprise) ────────────────────────
            from src.llm_factory import get_combined_model_list

            model_entries = []
            if st.session_state.sql_agent:
                try:
                    model_entries = get_combined_model_list(
                        st.session_state.sql_agent.ollama_manager
                    )
                except Exception:
                    pass

            if not model_entries:
                try:
                    from src.ollama_llm import OllamaManager
                    temp_manager = OllamaManager(CONFIG_PATH)
                    model_entries = get_combined_model_list(temp_manager)
                except Exception:
                    pass

            # Fallback: current ollama model as plain entry
            current_model = ollama_config.get('model', 'sqlcoder:7b')
            if not model_entries:
                model_entries = [{
                    "provider": "ollama",
                    "model": current_model,
                    "label": f"Ollama – {current_model}",
                }]

            labels = [e["label"] for e in model_entries]

            # Determine current selection index
            cur_provider = st.session_state.llm_provider
            cur_model = st.session_state.llm_model or current_model
            sel_index = 0
            for i, e in enumerate(model_entries):
                if e["provider"] == cur_provider and e["model"] == cur_model:
                    sel_index = i
                    break

            selected_label = st.selectbox(
                "Model",
                options=labels,
                index=sel_index,
                key='unified_model_label',
                help="Ollama models are local; Enterprise models use ENTERPRISE_LLM_* env vars."
            )

            # Store provider + model in session state when selection changes
            for e in model_entries:
                if e["label"] == selected_label:
                    st.session_state.llm_provider = e["provider"]
                    st.session_state.llm_model = e["model"]
                    # Keep ollama_model in sync for Save Settings / legacy path
                    if e["provider"] == "ollama":
                        st.session_state.ollama_model = e["model"]
                    break

            # ── Agent Mode selector ─────────────────────────────────────────────
            st.markdown("**🔁 Agent Mode**")
            mode_options = ["multi", "single"]
            mode_labels = ["Multi-Agent (default)", "Single-Agent"]
            cur_mode_idx = 0 if st.session_state.agent_mode != "single" else 1
            selected_mode_label = st.radio(
                "Agent Mode",
                options=mode_labels,
                index=cur_mode_idx,
                key="agent_mode_radio",
                label_visibility="collapsed",
                help="Multi-Agent: Analyst→Expert→Developer pipeline. Single-Agent: one unified call."
            )
            st.session_state.agent_mode = "single" if selected_mode_label == "Single-Agent" else "multi"

            if 'temperature' not in st.session_state:
                st.session_state.temperature = ui_settings.get('temperature', 0.7)
            if 'max_tokens' not in st.session_state:
                st.session_state.max_tokens = ui_settings.get('max_tokens', 2048)

            if st.button("Test Ollama Connection"):
                with st.spinner("Testing..."):
                    try:
                        save_current_settings()
                        if st.session_state.sql_agent is None:
                            st.session_state.sql_agent = SQLAgent(CONFIG_PATH)
                        if st.session_state.sql_agent.test_ollama_connection():
                            st.session_state.ollama_connected = True
                            st.success("Ollama Connected!")
                        else:
                            st.error("Connection failed")
                    except Exception as e:
                        st.error(f"Error: {str(e)}")


        st.markdown("---")
        if st.button("💾 Save Settings", use_container_width=True):
            save_current_settings()


def visualize_data(df: pd.DataFrame):
    """Automatically visualize data based on column types."""
    if df.empty or len(df) < 2:
        return

    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object', 'category', 'string']).columns.tolist()
    date_cols = df.select_dtypes(include=['datetime']).columns.tolist()

    st.markdown("### Visualization")
    try:
        if len(date_cols) >= 1 and len(numeric_cols) >= 1:
            fig = px.line(df, x=date_cols[0], y=numeric_cols[0], title=f"{numeric_cols[0]} over Time")
            st.plotly_chart(fig, use_container_width=True)
        elif len(categorical_cols) >= 1 and len(numeric_cols) >= 1:
            chart_df = df.head(20) if len(df) > 20 else df
            if len(df) > 20:
                st.caption("Showing top 20 rows")
            fig = px.bar(chart_df, x=categorical_cols[0], y=numeric_cols[0],
                         title=f"{numeric_cols[0]} by {categorical_cols[0]}")
            st.plotly_chart(fig, use_container_width=True)
        elif len(numeric_cols) >= 2:
            fig = px.scatter(df, x=numeric_cols[0], y=numeric_cols[1],
                             title=f"{numeric_cols[1]} vs {numeric_cols[0]}")
            st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.warning(f"Could not generate chart: {str(e)}")


def handle_feedback(query, sql, rating):
    """Handle feedback submission."""
    if st.session_state.db_connected and st.session_state.sql_agent:
        if save_feedback_to_db(st.session_state.sql_agent.db_manager, query, sql, rating):
            st.toast(f"Feedback saved ({rating})")
        else:
            st.error("Failed to save feedback")
    else:
        st.warning("Connect to DB to save feedback")


def render_chat_messages(user_id: str):
    """Render the conversation history for the current user."""
    messages = st.session_state.messages.get(user_id, [])

    for i, message in enumerate(messages):
        with st.chat_message(message["role"]):
            # Show modification banner if this was a modification turn
            if message.get("is_modification"):
                st.markdown(
                    '<div class="modification-banner">🔄 Modifying previous query...</div>',
                    unsafe_allow_html=True
                )

            st.markdown(message["content"])

            # Show SQL block if present
            if "sql" in message:
                with st.expander("🔍 View SQL", expanded=(i == len(messages) - 1)):
                    st.code(message["sql"], language="sql")

            # Show results if present
            if "results" in message:
                st.dataframe(message["results"])
                visualize_data(message["results"])

            # Run Query button if SQL exists but no results
            if "sql" in message and "results" not in message:
                if st.button("▶️ Run Query", key=f"run_{i}"):
                    with st.spinner("Executing query..."):
                        if st.session_state.sql_agent:
                            exec_res = st.session_state.sql_agent.execute_sql(message["sql"])
                            if exec_res["success"]:
                                st.session_state.messages[user_id][i]["results"] = pd.DataFrame(exec_res["data"])
                                st.rerun()
                            else:
                                st.error(f"Execution failed: {exec_res.get('error')}")

            # Response time
            if "time_taken" in message:
                st.markdown(
                    f'<p class="response-time">⏱ Response time: {message["time_taken"]}</p>',
                    unsafe_allow_html=True
                )

            # Feedback buttons for assistant messages
            if message["role"] == "assistant" and "sql" in message:
                st.markdown("---")
                st.markdown("**Rate this response:**")
                col1, col2, col3 = st.columns([1, 1, 8])
                with col1:
                    if st.button("👍", key=f"up_{i}", help="Helpful"):
                        user_query = messages[i - 1]["content"] if i > 0 else ""
                        handle_feedback(user_query, message["sql"], "positive")
                with col2:
                    if st.button("👎", key=f"down_{i}", help="Not Helpful"):
                        user_query = messages[i - 1]["content"] if i > 0 else ""
                        handle_feedback(user_query, message["sql"], "negative")
                with col3:
                    st.caption("💬 Need changes? Just type them below!")


def main():
    initialize_session_state()
    sidebar_settings()

    # Main Chat Interface
    st.title("💬 SQL Assistant")

    # ── Block main app until user_id is set ──────────────────────────────
    if not st.session_state.user_authenticated:
        st.warning("⚠️ Please identify yourself in the sidebar to start chatting")
        st.info(
            "👈 Enter your **Employee ID** (e.g. `12345`) or **Mobile Number** "
            "in the sidebar and click **Start Session**.\n\n"
            "Your conversation is private and scoped to your identity."
        )
        # Show demo info
        with st.expander("ℹ️ How conversational memory works"):
            st.markdown("""
            **Iterative SQL refinement:**
            1. Ask any data question → get SQL
            2. Say *"fix the JOIN to use customer_no"* → agent edits the exact SQL
            3. Say *"add WHERE amount > 50000"* → filter appended to Step 2's SQL
            4. Each user has **isolated memory** — corrections for one user never bleed into another
            5. Full-text search retrieves relevant context even from 20+ turns back
            """)
        return

    user_id = st.session_state.user_id

    # ── Conversation header ──────────────────────────────────────────────
    msg_count = len(st.session_state.messages.get(user_id, []))
    col1, col2 = st.columns([3, 1])
    with col1:
        st.markdown(
            f'Session: <span class="user-id-badge">🔑 {user_id}</span>&nbsp;&nbsp;'
            f'<span style="color:#666;font-size:0.85rem">{msg_count} messages</span>',
            unsafe_allow_html=True
        )
    with col2:
        with st.expander("💡 Tips"):
            st.markdown("""
            **Conversational tips:**
            - Ask new questions naturally
            - *"fix the join"* → edits last SQL
            - *"add WHERE amount > 50000"* → adds filter
            - *"change GROUP BY to week"* → adjusts grouping
            - Rate responses with 👍 or 👎
            """)

    # ── Agent mode badge ──────────────────────────────────────────────────────
    _mode = st.session_state.get("agent_mode") or get_effective_mode()
    _provider = st.session_state.get("llm_provider", "ollama")
    _mode_label = "🔹 Single-Agent" if _mode == "single" else "🔷 Multi-Agent"
    _provider_label = "Enterprise" if _provider == "enterprise" else "Ollama"
    st.caption(f"Mode: **{_mode_label}** · Provider: **{_provider_label}** (`SQL_AGENT_MODE={_mode}`)")



    # ── Display chat messages ────────────────────────────────────────────
    render_chat_messages(user_id)

    # ── Chat Input ───────────────────────────────────────────────────────
    if prompt := st.chat_input("Ask a question about your data..."):
        # Ensure user message list exists
        if user_id not in st.session_state.messages:
            st.session_state.messages[user_id] = []

        # Add user message to UI history
        st.session_state.messages[user_id].append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        if not st.session_state.db_connected:
            st.error("Please connect to the database first via settings.")
            return

        # ── Generate response ────────────────────────────────────────────
        with st.chat_message("assistant"):
            with st.status("Processing query...", expanded=True) as status:
                start_time = time.time()
                try:
                    # Update agent LLM settings
                    if st.session_state.sql_agent:
                        st.session_state.sql_agent.ollama_manager.llm.temperature = st.session_state.temperature
                        st.session_state.sql_agent.ollama_manager.llm.max_tokens = st.session_state.max_tokens

                    status.write("🧠 Analyzing query and retrieving context...")

                    # Detect if this looks like a modification
                    pg_mem = getattr(st.session_state.sql_agent, 'pg_memory', None)
                    feedback_tag = None
                    is_modification = False
                    if pg_mem:
                        feedback_tag = pg_mem.extract_feedback(prompt)
                        is_modification = feedback_tag is not None

                    if is_modification:
                        status.write(f"🔄 Modification detected: `{feedback_tag}` — retrieving prior SQL...")

                    # Call generate_sql (memory saving handled inside sql_agent)
                    result = st.session_state.sql_agent.generate_sql(prompt)

                    time_taken = format_time_taken(start_time)

                    if result["success"]:
                        status.update(label="✅ SQL Generated!", state="complete", expanded=False)

                        sql_query = result["sql_query"]
                        response_content = ""
                        if result.get("cached"):
                            response_content = "⚡ (From Cache)"
                        elif is_modification:
                            response_content = f"🔄 Applied modification: `{feedback_tag}`"

                        if response_content:
                            st.markdown(response_content)

                        with st.expander("🔍 View SQL", expanded=True):
                            st.code(sql_query, language="sql")

                        st.markdown(
                            f'<p class="response-time">⏱ Response time: {time_taken}</p>',
                            unsafe_allow_html=True
                        )

                        # Save to UI history
                        message_data = {
                            "role": "assistant",
                            "content": response_content or "SQL generated successfully.",
                            "sql": sql_query,
                            "time_taken": time_taken,
                            "is_modification": is_modification
                        }
                        st.session_state.messages[user_id].append(message_data)
                        st.rerun()

                    else:
                        status.update(label="❌ Generation Failed", state="error")
                        error_msg = f"Failed to generate SQL: {result.get('error')}"
                        st.error(error_msg)
                        st.session_state.messages[user_id].append({
                            "role": "assistant",
                            "content": error_msg,
                            "time_taken": time_taken
                        })

                except Exception as e:
                    status.update(label="❌ Error Occurred", state="error")
                    st.error(f"An error occurred: {str(e)}")


if __name__ == "__main__":
    main()
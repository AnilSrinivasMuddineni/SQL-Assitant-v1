__import__("pysqlite3")
import sys
sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")

import streamlit as st
import pandas as pd
import time
import plotly.express as px
import os

# Set dummy OpenAI API key to bypass CrewAI/LangChain strict validation
# This is required even when using local Ollama models via ChatOpenAI
os.environ["OPENAI_API_KEY"] = "NA"
os.environ["OPENAI_API_BASE"] = "http://localhost:11434/v1"
os.environ["OPENAI_BASE_URL"] = "http://localhost:11434/v1"
os.environ["OPENAI_MODEL_NAME"] = "sqlcoder:7b" # Default model, will be updated dynamically

from src.sql_agent import SQLAgent
from src.utils import load_config, save_config, format_time_taken, load_settings_from_db, save_settings_to_db, init_settings_table, save_feedback_to_db

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
</style>
""", unsafe_allow_html=True)

def initialize_session_state():
    """Initialize session state variables."""
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    if 'sql_agent' not in st.session_state:
        st.session_state.sql_agent = None
    if 'db_connected' not in st.session_state:
        st.session_state.db_connected = False
    if 'ollama_connected' not in st.session_state:
        st.session_state.ollama_connected = False
    
    # Load local config for DB connection details
    if 'config' not in st.session_state:
        st.session_state.config = load_config(CONFIG_PATH)

    # Try to connect to DB and load remote settings if possible
    if not st.session_state.db_connected and st.session_state.sql_agent is None:
        try:
            agent = SQLAgent(CONFIG_PATH)
            
            # Connect to DB first (without automatically loading everything via connect_database)
            if agent.db_manager.connect():
                st.session_state.sql_agent = agent
                st.session_state.db_connected = True
                
                # Attempt to load schema using the config-driven strategy first
                logger_msg = "Attempting to load schema from configuration..."
                print(logger_msg) # Ensure visibility in logs
                
                if not agent.load_configured_schema():
                    # Fallback: Load all DDLs if config loading failed or returned no tables
                    print("Config loading failed or empty. Falling back to loading ALL DDLs.")
                    ddls = agent.db_manager.get_all_ddls()
                    agent.vector_store.store_ddls(ddls)
                
                # Ensure table exists
                
                # Ensure table exists
                init_settings_table(agent.db_manager)
                
                # Load settings from DB
                db_settings = load_settings_from_db(agent.db_manager)
                if db_settings:
                    # Merge DB settings into session config
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
    
    # Update Ollama settings
    new_config['ollama']['base_url'] = st.session_state.ollama_url
    new_config['ollama']['model'] = st.session_state.ollama_model
    
    # Update UI settings
    if 'settings' not in new_config:
        new_config['settings'] = {}
    new_config['settings']['temperature'] = st.session_state.temperature
    new_config['settings']['max_tokens'] = st.session_state.max_tokens
    
    # 1. Save DB connection details locally (Preserve existing DB config)
    if save_config(CONFIG_PATH, new_config):
        st.session_state.config = new_config
        st.toast("Local config saved")
    
    # 2. Save Preferences to Database if connected
    if st.session_state.db_connected and st.session_state.sql_agent:
        db_settings = {
            "ollama": new_config['ollama'],
            "settings": new_config['settings']
        }
        if save_settings_to_db(st.session_state.sql_agent.db_manager, db_settings):
            st.toast("Preferences saved to Database")
            
            # Update the running agent with new model settings
            st.session_state.sql_agent.update_model(
                model_name=new_config['ollama']['model'],
                base_url=new_config['ollama']['base_url']
            )
        else:
            st.error("Failed to save to Database")
    else:
        st.warning("Connect to Database to save preferences remotely")

def sidebar_settings():
    """Render the settings sidebar."""
    with st.sidebar:
        st.title(" Settings")
        
        # Connection Status
        if st.session_state.db_connected:
            st.success(" Database Connected")
        else:
            st.error(" Database Disconnected")
            st.info("Check config/database_config.json")
            if st.button("Retry Connection"):
                st.rerun()

        # Model Settings
        with st.expander(" Model Configuration", expanded=True):
            ollama_config = st.session_state.config.get('ollama', {})
            ui_settings = st.session_state.config.get('settings', {})
            
            st.text_input("Base URL", value=ollama_config.get('base_url', 'http://localhost:11434'), key='ollama_url')
            
            # Fetch available models
            available_models = []
            if st.session_state.sql_agent:
                available_models = st.session_state.sql_agent.ollama_manager.get_available_models()
            
            # Fallback if agent not initialized or fetch failed
            if not available_models:
                # Try to create a temporary manager to fetch models if agent isn't ready
                try:
                    from src.ollama_llm import OllamaManager
                    temp_manager = OllamaManager(CONFIG_PATH)
                    available_models = temp_manager.get_available_models()
                except:
                    pass

            current_model = ollama_config.get('model', 'sqlcoder:7b')
            
            if available_models:
                # Ensure current model is in the list
                if current_model not in available_models:
                    available_models.append(current_model)
                
                index = available_models.index(current_model)
                st.selectbox("Model", options=available_models, index=index, key='ollama_model')
            else:
                st.text_input("Model", value=current_model, key='ollama_model', help="Could not fetch models from Ollama")
            
            # Ensure defaults are in session state since widgets are hidden
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

        # Save Button
        st.markdown("---")
        if st.button(" Save Settings", use_container_width=True):
            save_current_settings()

def visualize_data(df: pd.DataFrame):
    """
    Automatically visualize data based on column types.
    """
    if df.empty or len(df) < 2:
        return

    # Identify column types
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object', 'category', 'string']).columns.tolist()
    date_cols = df.select_dtypes(include=['datetime']).columns.tolist()

    st.markdown("### Visualization")

    try:
        # Case 1: Time Series (1 Date + 1 Numeric)
        if len(date_cols) >= 1 and len(numeric_cols) >= 1:
            fig = px.line(df, x=date_cols[0], y=numeric_cols[0], title=f"{numeric_cols[0]} over Time")
            st.plotly_chart(fig, use_container_width=True)
        
        # Case 2: Bar Chart (1 Categorical + 1 Numeric)
        elif len(categorical_cols) >= 1 and len(numeric_cols) >= 1:
            # Limit to top 20 for readability
            if len(df) > 20:
                chart_df = df.head(20)
                st.caption("Showing top 20 rows")
            else:
                chart_df = df
            
            fig = px.bar(chart_df, x=categorical_cols[0], y=numeric_cols[0], title=f"{numeric_cols[0]} by {categorical_cols[0]}")
            st.plotly_chart(fig, use_container_width=True)
            
        # Case 3: Scatter Plot (2 Numeric)
        elif len(numeric_cols) >= 2:
            fig = px.scatter(df, x=numeric_cols[0], y=numeric_cols[1], title=f"{numeric_cols[1]} vs {numeric_cols[0]}")
            st.plotly_chart(fig, use_container_width=True)
            
        # Case 4: Pie Chart (1 Categorical + 1 Numeric, small dataset)
        elif len(categorical_cols) >= 1 and len(numeric_cols) >= 1 and len(df) <= 10:
            fig = px.pie(df, names=categorical_cols[0], values=numeric_cols[0], title=f"Distribution of {numeric_cols[0]}")
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

def main():
    initialize_session_state()
    sidebar_settings()
    
    # Main Chat Interface
    st.title("SQL Assistant")
    
    # Display chat messages
    for i, message in enumerate(st.session_state.messages):
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if "sql" in message:
                st.code(message["sql"], language="sql")
            
            if "results" in message:
                st.dataframe(message["results"])
                visualize_data(message["results"])
            
            # Add Run Query button if SQL exists but no results yet
            if "sql" in message and "results" not in message:
                if st.button("Run Query", key=f"run_{i}"):
                    with st.spinner("Executing query..."):
                        if st.session_state.sql_agent:
                            exec_res = st.session_state.sql_agent.execute_sql(message["sql"])
                            if exec_res["success"]:
                                # Update message with results
                                st.session_state.messages[i]["results"] = pd.DataFrame(exec_res["data"])
                                st.session_state.messages[i]["time_taken"] += f" + {format_time_taken(time.time())} (exec)" # Approximate
                                st.rerun()
                            else:
                                st.error(f"Execution failed: {exec_res.get('error')}")

            if "time_taken" in message:
                st.markdown(f'<p class="response-time"> Response time: {message["time_taken"]}</p>', unsafe_allow_html=True)
            
            # Add feedback buttons for assistant messages
            if message["role"] == "assistant" and "sql" in message:
                col1, col2 = st.columns([1, 15])
                with col1:
                    if st.button("👍", key=f"up_{i}", help="Helpful"):
                        # Find the corresponding user query (usually the message before)
                        user_query = ""
                        if i > 0 and st.session_state.messages[i-1]["role"] == "user":
                            user_query = st.session_state.messages[i-1]["content"]
                        handle_feedback(user_query, message["sql"], "positive")
                with col2:
                    if st.button("👎", key=f"down_{i}", help="Not Helpful"):
                        user_query = ""
                        if i > 0 and st.session_state.messages[i-1]["role"] == "user":
                            user_query = st.session_state.messages[i-1]["content"]
                        handle_feedback(user_query, message["sql"], "negative")

    # Chat Input
    if prompt := st.chat_input("Ask a question about your data..."):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Check connections
        if not st.session_state.db_connected:
            st.error("Please connect to the database first via settings.")
            return
        
        # Generate response
        with st.chat_message("assistant"):
            # Create a status container
            with st.status("Processing query...", expanded=True) as status:
                start_time = time.time()
                try:
                    # Update agent settings
                    if st.session_state.sql_agent:
                        st.session_state.sql_agent.ollama_manager.llm.temperature = st.session_state.temperature
                        st.session_state.sql_agent.ollama_manager.llm.max_tokens = st.session_state.max_tokens

                    # 1. Retrieve Context
                    status.write("Processing query...")
                    
                    result = st.session_state.sql_agent.generate_sql(prompt)
                    
                   # status.write("🧠 Analyzing requirements and generating SQL...")
                    
                    time_taken = format_time_taken(start_time)
                    
                    if result["success"]:
                        status.update(label="SQL Generated!", state="complete", expanded=False)
                        
                        response_content = ""
                        if result.get("cached"):
                            response_content += " ( From Cache)"
                        
                        sql_query = result["sql_query"]
                        
                        if response_content:
                            st.markdown(response_content)
                        st.code(sql_query, language="sql")
                        
                        # Execution is now manual via the button in the message loop
                        execution_results = None
                        
                        st.markdown(f'<p class="response-time"> - Response time: {time_taken}</p>', unsafe_allow_html=True)
                        
                        # Save to history
                        message_data = {
                            "role": "assistant",
                            "content": response_content,
                            "sql": sql_query,
                            "time_taken": time_taken
                        }
                        # We don't add results here anymore, they are added upon execution
                        
                        st.session_state.messages.append(message_data)
                        
                        # Force rerun to show the new message with the Run button
                        st.rerun()
                        
                    else:
                        status.update(label="Generation Failed", state="error")
                        error_msg = f"Failed to generate SQL: {result.get('error')}"
                        st.error(error_msg)
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": error_msg,
                            "time_taken": time_taken
                        })
                        
                except Exception as e:
                    status.update(label="Error Occurred", state="error")
                    st.error(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()
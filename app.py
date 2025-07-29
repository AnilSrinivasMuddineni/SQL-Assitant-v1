import streamlit as st
import pandas as pd
import json
import time
from typing import Dict, Any
import sys
import os

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from sql_agent import SQLAgent

# Page configuration
st.set_page_config(
    page_title="CrewAI SQL Agent",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .status-success {
        color: #28a745;
        font-weight: bold;
    }
    .status-error {
        color: #dc3545;
        font-weight: bold;
    }
    .status-warning {
        color: #ffc107;
        font-weight: bold;
    }
    .query-box {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #dee2e6;
    }
    .result-box {
        background-color: #e9ecef;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #ced4da;
        font-family: 'Courier New', monospace;
    }
</style>
""", unsafe_allow_html=True)

def initialize_session_state():
    """Initialize session state variables."""
    if 'sql_agent' not in st.session_state:
        st.session_state.sql_agent = None
    if 'db_connected' not in st.session_state:
        st.session_state.db_connected = False
    if 'ollama_connected' not in st.session_state:
        st.session_state.ollama_connected = False
    if 'query_history' not in st.session_state:
        st.session_state.query_history = []

def main():
    """Main application function."""
    
    # Initialize session state
    initialize_session_state()
    
    # Header
    st.markdown('<h1 class="main-header">🤖 CrewAI SQL Agent</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Natural Language to SQL Query Generator</p>', unsafe_allow_html=True)
    
    # Sidebar for configuration and status
    with st.sidebar:
        st.header("🔧 Configuration")
        
        # Connection status
        st.subheader("Connection Status")
        
        # Database connection
        if st.button("🔌 Connect to Database", key="db_connect"):
            with st.spinner("Connecting to database..."):
                try:
                    if st.session_state.sql_agent is None:
                        st.session_state.sql_agent = SQLAgent()
                    
                    if st.session_state.sql_agent.connect_database():
                        st.session_state.db_connected = True
                        st.success("✅ Database connected!")
                    else:
                        st.error("❌ Database connection failed!")
                except Exception as e:
                    st.error(f"❌ Database connection error: {str(e)}")
        
        if st.session_state.db_connected:
            st.markdown('<p class="status-success">✅ Database: Connected</p>', unsafe_allow_html=True)
        else:
            st.markdown('<p class="status-error">❌ Database: Disconnected</p>', unsafe_allow_html=True)
        
        # Ollama connection
        if st.button("🤖 Test Ollama Connection", key="ollama_test"):
            with st.spinner("Testing Ollama connection..."):
                try:
                    if st.session_state.sql_agent is None:
                        st.session_state.sql_agent = SQLAgent()
                    
                    if st.session_state.sql_agent.test_ollama_connection():
                        st.session_state.ollama_connected = True
                        st.success("✅ Ollama connected!")
                    else:
                        st.error("❌ Ollama connection failed!")
                except Exception as e:
                    st.error(f"❌ Ollama connection error: {str(e)}")
        
        if st.session_state.ollama_connected:
            st.markdown('<p class="status-success">✅ Ollama: Connected</p>', unsafe_allow_html=True)
        else:
            st.markdown('<p class="status-warning">⚠️ Ollama: Not tested</p>', unsafe_allow_html=True)
        
        # Configuration
        st.subheader("⚙️ Settings")
        
        # Model settings
        st.selectbox(
            "Model Temperature",
            [0.1, 0.3, 0.5, 0.7, 0.9],
            index=3,
            key="temperature"
        )
        
        st.selectbox(
            "Max Tokens",
            [1024, 2048, 4096, 8192],
            index=1,
            key="max_tokens"
        )
        
        # Query history
        st.subheader("📚 Query History")
        if st.session_state.query_history:
            for i, (query, sql) in enumerate(st.session_state.query_history[-5:]):
                with st.expander(f"Query {i+1}: {query[:50]}..."):
                    st.text_area("Natural Language", query, disabled=True, key=f"hist_query_{i}")
                    st.text_area("Generated SQL", sql, disabled=True, key=f"hist_sql_{i}")
        else:
            st.info("No query history yet.")
    
    # Main content area
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("💬 Natural Language Query")
        
        # Query input
        natural_language_query = st.text_area(
            "Enter your query in natural language:",
            placeholder="e.g., Show me all users who registered in the last 30 days",
            height=150,
            key="natural_query"
        )
        
        # Generate SQL button
        if st.button("🚀 Generate SQL", type="primary", key="generate_sql"):
            if not natural_language_query.strip():
                st.error("Please enter a natural language query.")
            elif not st.session_state.db_connected:
                st.error("Please connect to the database first.")
            elif not st.session_state.ollama_connected:
                st.error("Please test Ollama connection first.")
            else:
                with st.spinner("Generating SQL query..."):
                    try:
                        # Update model parameters
                        if st.session_state.sql_agent:
                            st.session_state.sql_agent.ollama_manager.llm.temperature = st.session_state.temperature
                            st.session_state.sql_agent.ollama_manager.llm.max_tokens = st.session_state.max_tokens
                        
                        # Generate SQL
                        result = st.session_state.sql_agent.generate_sql(natural_language_query)
                        
                        if result["success"]:
                            # Store in session state
                            st.session_state.generated_sql = result["sql_query"]
                            st.session_state.query_result = result
                            
                            # Add to history
                            st.session_state.query_history.append((
                                natural_language_query,
                                result["sql_query"]
                            ))
                            
                            st.success("✅ SQL generated successfully!")
                        else:
                            st.error(f"❌ SQL generation failed: {result.get('error', 'Unknown error')}")
                            
                    except Exception as e:
                        st.error(f"❌ Error: {str(e)}")
    
    with col2:
        st.header("📝 Generated SQL")
        
        if 'generated_sql' in st.session_state:
            st.markdown('<div class="query-box">', unsafe_allow_html=True)
            st.code(st.session_state.generated_sql, language="sql")
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Execute SQL button
            if st.button("▶️ Execute SQL", key="execute_sql"):
                with st.spinner("Executing SQL query..."):
                    try:
                        execution_result = st.session_state.sql_agent.execute_sql(st.session_state.generated_sql)
                        
                        if execution_result["success"]:
                            st.session_state.execution_result = execution_result
                            st.success(f"✅ Query executed successfully! {execution_result['row_count']} rows returned.")
                        else:
                            st.error(f"❌ Query execution failed: {execution_result.get('error', 'Unknown error')}")
                            
                    except Exception as e:
                        st.error(f"❌ Execution error: {str(e)}")
        else:
            st.info("Generate a SQL query to see it here.")
    
    # Results section
    if 'execution_result' in st.session_state and st.session_state.execution_result["success"]:
        st.header("📊 Query Results")
        
        result = st.session_state.execution_result
        
        # Display results in a table
        if result["data"]:
            df = pd.DataFrame(result["data"])
            st.dataframe(df, use_container_width=True)
            
            # Download button
            csv = df.to_csv(index=False)
            st.download_button(
                label="📥 Download Results as CSV",
                data=csv,
                file_name="query_results.csv",
                mime="text/csv"
            )
        else:
            st.info("Query returned no results.")
    
    # Debug information (expandable)
    with st.expander("🔍 Debug Information"):
        if 'query_result' in st.session_state:
            st.subheader("Query Analysis")
            result = st.session_state.query_result
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Relevant Tables:**")
                st.write(result.get("relevant_tables", []))
                
                st.write("**Schema Context:**")
                st.text_area("Schema", result.get("schema_context", ""), height=200, disabled=True)
            
            with col2:
                st.write("**Crew Result:**")
                st.text_area("Crew Output", result.get("crew_result", ""), height=300, disabled=True)

if __name__ == "__main__":
    main() 
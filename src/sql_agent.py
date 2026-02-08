import json
import logging
import re
import hashlib
import pandas as pd
from io import BytesIO
from typing import List, Dict, Any, Optional
from crewai import Agent, Task, Crew, Process
from langchain_community.llms import Ollama
from crewai.tools import tool
from langchain_ollama import OllamaLLM

from src.database_manager import DatabaseManager
from src.ollama_llm import OllamaManager
from src.vector_store import VectorStore
from src.utils import get_cached_query, cache_query

logger = logging.getLogger(__name__)

class SQLAgent:
    def __init__(self, config_path: str = "config/database_config.json", 
                 user_id: Optional[int] = None, session_id: Optional[str] = None):
        """
        Initialize SQL Agent with all components.
        
        Args:
            config_path: Path to configuration file
            user_id: Optional user ID for memory management
            session_id: Optional session ID for memory management
        """
        self.config_path = config_path
        self.db_manager = DatabaseManager(config_path)
        self.ollama_manager = OllamaManager(config_path)
        self.vector_store = VectorStore()
        self.llm = self.ollama_manager.llm
        self._setup_logging()
        
        # User and session tracking
        self.user_id = user_id
        self.session_id = session_id
        
        # Initialize memory manager if user_id and session_id are provided
        self.memory_manager = None
        if user_id and session_id:
            try:
                from src.memory_manager import MemoryManager
                config = self._load_config()
                chroma_path = config.get('memory', {}).get('chroma_path', './chroma_db')
                self.memory_manager = MemoryManager(
                    user_id=user_id,
                    session_id=session_id,
                    db_manager=self.db_manager,
                    chroma_path=chroma_path
                )
                logger.info(f"MemoryManager initialized for user {user_id}, session {session_id}")
            except Exception as e:
                logger.warning(f"Failed to initialize MemoryManager: {str(e)}")
                self.memory_manager = None
        
        # Load sample queries for context
        self.sample_queries = self._load_sample_queries()
        
        # Initialize agents
        self.agents = self._create_agents()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from file."""
        try:
            with open(self.config_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error loading config: {str(e)}")
            return {}

    def _setup_logging(self):
        try:
            with open(self.config_path, 'r') as f:
                config = json.load(f)
            logging_config = config.get("logging", {})

            level_str = logging_config.get("level", "INFO").upper()
            log_level = getattr(logging, level_str, logging.INFO)

            handlers = logging_config.get("handlers", ["console"])
            filename = logging_config.get("filename", "app.log")

            logger = logging.getLogger()
            logger.setLevel(log_level)

            # Remove existing handlers
            for handler in logger.handlers[:]:
                logger.removeHandler(handler)

            if "console" in handlers:
                ch = logging.StreamHandler()
                ch.setLevel(log_level)
                formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
                ch.setFormatter(formatter)
                logger.addHandler(ch)

            if "file" in handlers:
                fh = logging.FileHandler(filename)
                fh.setLevel(log_level)
                formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
                fh.setFormatter(formatter)
                logger.addHandler(fh)

            logger.info(f"Logging configured: level={level_str}, handlers={handlers}")

        except FileNotFoundError:
            logging.basicConfig(level=logging.INFO)
            logging.warning(f"Config file {self.config_path} not found. Using basic logging config.")
        except json.JSONDecodeError as e:
            logging.basicConfig(level=logging.INFO)
            logging.error(f"Invalid JSON in config file {self.config_path}: {str(e)}")
        except Exception as e:
            logging.basicConfig(level=logging.INFO)
            logging.error(f"Unexpected error during logging setup: {str(e)}")
        
    def _load_sample_queries(self) -> List[Dict[str, str]]:
        """Load sample queries from JSON file."""
        try:
            with open("data/sample_queries.json", 'r') as f:
                data = json.load(f)
                return data.get("queries", [])
        except FileNotFoundError:
            logger.warning("Sample queries file not found. Using empty list.")
            return []
        except json.JSONDecodeError:
            logger.error("Invalid JSON in sample queries file.")
            return []
    
    def _create_agents(self) -> Dict[str, Agent]:
        """Create CrewAI agents for different roles."""

        # Use the LLM from OllamaManager which is initialized with config
        self.llm = self.ollama_manager.llm
        logger.info(f"Using LLM: {type(self.llm)}")
        try:
            logger.info(f"LLM Model Name: {getattr(self.llm, 'model_name', 'Unknown')}")
        except:
            pass
        
        # SQL Analyst Agent
        sql_analyst = Agent(
            role="SQL Analyst",
            goal="Analyze natural language queries and understand database requirements",
            backstory="""You are an expert SQL analyst with years of experience in 
            database design and query optimization. You excel at understanding user 
            requirements and translating them into database operations.""",
            verbose=True,
            allow_delegation=False,
            llm=self.llm
        )
        
        # Database Expert Agent
        db_expert = Agent(
            role="Database Expert",
            goal="Understand database schema and provide context for SQL generation",
            backstory="""You are a database expert who knows PostgreSQL inside and out. 
            You understand table relationships, data types, and can provide valuable 
            context about the database structure.""",
            verbose=True,
            allow_delegation=False,
            llm=self.llm
        )
        
        # SQL Developer Agent
        sql_developer = Agent(
            role="SQL Developer",
            goal="Generate accurate and efficient PostgreSQL queries",
            backstory="""You are a senior SQL developer who writes clean, efficient, 
            and correct PostgreSQL queries. You follow best practices and ensure 
            queries are optimized for performance.""",
            verbose=True,
            allow_delegation=False,
            llm=self.llm
        )
        
        return {
            "sql_analyst": sql_analyst,
            "db_expert": db_expert,
            "sql_developer": sql_developer
        }
    
    def connect_database(self) -> bool:
        """Connect to the database and initialize vector store."""
        if self.db_manager.connect():
            try:
                # Fetch all DDLs and store in vector DB
                logger.info("Fetching DDLs for Vector Store...")
                ddls = self.db_manager.get_all_ddls()
                
                # Clear existing store to promote fresh state
                self.vector_store.clear_store()
                self.vector_store.store_ddls(ddls)
                return True
            except Exception as e:
                logger.error(f"Error initializing Vector Store: {str(e)}")
                # We still return True if DB connects, even if Vector Store fails
                return True
        return False

    def load_configured_schema(self, config_table_name: Optional[str] = None) -> bool:
        """
        Load schema based on a configuration table instead of the entire database.
        
        Fetches DDLs for tables specified in the config table (defined in database_config.json
        or overridden via argument).
        
        Args:
            config_table_name: Optional override for the config table name.
            
        Returns:
            bool: True if successful, False otherwise.
        """
        if not self.db_manager.engine:
             logger.warning("Database not connected. Attempting to connect...")
             if not self.db_manager.connect():
                 logger.error("Failed to connect to database.")
                 return False

        try:
            # We pass the argument (even if None) to the manager, which handles the config fallback
            ddls = self.db_manager.get_configured_ddls(config_table_name=config_table_name)
            
            if ddls:
                # Clear existing store to ensure we only have the configured tables
                logger.info("Clearing Vector Store for configured schema load...")
                self.vector_store.clear_store()
                
                self.vector_store.store_ddls(ddls)
                logger.info("Successfully loaded configured schema into Vector Store.")
                return True
            else:
                logger.warning("No DDLs were retrieved from configuration.")
                return False

        except Exception as e:
            logger.error(f"Error in load_configured_schema: {str(e)}")
            return False
    
    def test_ollama_connection(self) -> bool:
        """Test connection to Ollama service."""
        return self.ollama_manager.test_connection()

    def update_model(self, model_name: str, base_url: str):
        """Update the LLM model and recreate agents."""
        self.ollama_manager.update_model(model_name, base_url)
        self.llm = self.ollama_manager.llm
        self.agents = self._create_agents()
        logger.info(f"SQLAgent updated to use model: {model_name}")
    
    def _create_schema_context(self, relevant_ddls: List[str]) -> str:
        """Create schema context from retrieved DDLs."""
        if not relevant_ddls:
            return "No relevant tables found."
        
        return "\n\n".join(relevant_ddls)
    
    def _create_examples_context(self) -> str:
        """Create examples context from sample queries."""
        if not self.sample_queries:
            return "No example queries available."
        
        examples = []
        for i, query_info in enumerate(self.sample_queries[:5], 1):  # Limit to 5 examples
            examples.append(f"Example {i}:")
            examples.append(f"User: {query_info['natural_language']}")
            examples.append(f"SQL: {query_info['sql']}")
            examples.append("---")
            
        return "\n".join(examples)
    
    def prepare_enhanced_context(self, query: str, feedback: Optional[str] = None, 
                                use_enhanced_ddl: bool = True) -> Dict[str, str]:
        """
        Prepare enhanced context with conversation history, schema with comments, and feedback.
        
        Args:
            query: Current user query
            feedback: Optional user feedback for corrections
            use_enhanced_ddl: Whether to use enhanced DDLs with column comments
            
        Returns:
            Dictionary with formatted context components
        """
        from src.prompt_templates import format_conversation_history, format_user_feedback
        
        # Get conversation history if memory manager is available
        conversation_history = "No previous conversation."
        if self.memory_manager:
            conversation_history = self.memory_manager.format_for_prompt(limit=10)
        
        # Get relevant DDLs using RAG
        relevant_table_ddls = self.vector_store.retrieve_relevant_ddls(query)
        
        # Get enhanced DDLs with comments if requested
        schema_with_comments = ""
        if use_enhanced_ddl and relevant_table_ddls:
            try:
                # Extract table names from DDL strings and get enhanced versions
                enhanced_ddls = []
                for ddl in relevant_table_ddls:
                    # Try to extract table name from DDL
                    # This is a simple approach - could be improved
                    if "CREATE TABLE" in ddl:
                        # Parse table name
                        parts = ddl.split("CREATE TABLE")
                        if len(parts) > 1:
                            table_part = parts[1].strip().split("(")[0].strip()
                            # Handle schema.table format
                            if "." in table_part:
                                schema_name, table_name = table_part.split(".", 1)
                                table_name = table_name.strip()
                            else:
                                schema_name = None
                                table_name = table_part
                            
                            # Get enhanced DDL
                            enhanced_ddl = self.db_manager.get_enhanced_ddl(table_name, schema_name)
                            enhanced_ddls.append(enhanced_ddl)
                        else:
                            enhanced_ddls.append(ddl)
                    else:
                        enhanced_ddls.append(ddl)
                
                schema_with_comments = "\n\n".join(enhanced_ddls)
            except Exception as e:
                logger.warning(f"Error getting enhanced DDLs, using regular: {str(e)}")
                schema_with_comments = "\n\n".join(relevant_table_ddls)
        else:
            schema_with_comments = "\n\n".join(relevant_table_ddls) if relevant_table_ddls else "No relevant schema found."
        
        # Format feedback
        user_feedback = format_user_feedback(feedback) if feedback else "No feedback provided."
        
        return {
            "conversation_history": conversation_history,
            "schema_with_comments": schema_with_comments,
            "current_query": query,
            "user_feedback": user_feedback
        }

        if self.db_manager.engine:
            self.db_manager.engine.dispose()

    def _check_relevancy(self, query: str) -> bool:
        """
        Check if the query is relevant to SQL generation or database operations.
        
        Args:
           query: The user's input string.
           
        Returns:
           bool: True if relevant, False otherwise.
        """
        try:
             # Use the existing LLM instance
            prompt = f"""You are a strict classifier. Your job is to determine if a user's query is related to SQL, databases, data analysis, or asking for data from a system.
            
            Query: {query}
            
            Rules:
            - If it asks for code, data, tables, schema, inserting, updating, deleting, or analyzing numbers/text, answer YES.
            - If it is a greeting like "hi", "hello", answer NO.
            - If it asks about general knowledge (e.g. "capital of France", "recipe for cake"), answer NO.
            - If it is meaningless or random text, answer NO.
            
            Answer ONLY with the word YES or NO. Do not add punctuation or explanation.
            """
            
            response = self.llm.invoke(prompt)
            
            # Clean response
            clean_response = str(response).strip().upper()
            
            # Check for YES match
            if "YES" in clean_response:
                return True
            return False
            
        except Exception as e:
            logger.error(f"Error checking relevancy: {e}")
            # If check fails, fail open (allow it) or closed? 
            # Let's fail open to avoid blocking valid queries on LLM hiccups, 
            # but log it.
            return True

    def _classify_query_type(self, query: str, last_sql: Optional[str]) -> str:
        """
        Classify if query is a new question or modification of previous SQL.
        
        Args:
            query: Current user query
            last_sql: Last SQL generated in this session
            
        Returns:
            'NEW_QUERY' or 'MODIFICATION'
        """
        # If no last SQL, it must be new
        if not last_sql:
            logger.info("No last SQL found, classifying as NEW_QUERY")
            return 'NEW_QUERY'
        
        # Check for modification keywords
        modification_keywords = [
            'previous', 'last', 'above', 'that query', 'same query',
            'add', 'remove', 'fix', 'change', 'modify', 'update',
            'also include', 'instead', 'replace', 'adjust', 'correct',
            'use', 'switch', 'alter', 'edit'
        ]
        
        query_lower = query.lower()
        
        # Strong indicators of modification
        if any(keyword in query_lower for keyword in modification_keywords[:5]):
            logger.info(f"Detected modification keyword in query, classifying as MODIFICATION")
            return 'MODIFICATION'
        
        # Weaker indicators - use LLM for confirmation
        if any(keyword in query_lower for keyword in modification_keywords[5:]):
            logger.info("Detected potential modification keyword, using LLM classifier")
            try:
                from src.prompt_templates import QUERY_TYPE_CLASSIFIER
                
                prompt = QUERY_TYPE_CLASSIFIER.format(
                    user_message=query,
                    last_sql=last_sql[:200] if last_sql else "None"  # Truncate for context
                )
                
                response = self.llm.invoke(prompt).strip().upper()
                logger.info(f"LLM classification response: {response}")
                
                return 'MODIFICATION' if 'MODIFICATION' in response else 'NEW_QUERY'
            except Exception as e:
                logger.warning(f"Error in LLM classification: {str(e)}, defaulting to NEW_QUERY")
                return 'NEW_QUERY'
        
        # Default to new query
        logger.info("No modification keywords detected, classifying as NEW_QUERY")
        return 'NEW_QUERY'

    def generate_sql(self, natural_language_query: str, feedback: Optional[str] = None) -> Dict[str, Any]:
        """
        Generate SQL query from natural language using CrewAI with memory and feedback support.
        
        Args:
            natural_language_query: User's natural language query
            feedback: Optional feedback for query correction
            
        Returns:
            Dictionary with success status and generated SQL or error
        """
        try:
            # 0. Check Relevancy
            if not self._check_relevancy(natural_language_query):
                logger.info(f"Query flagged as irrelevant: {natural_language_query}")
                error_message = """I'm an SQL Assistant designed to help you query your database.

Please ask SQL-related questions like:
- 'Generate an SQL query to list all customers with high-value transactions in a given month'
- 'Generate SQL query to retrieve high-activity accounts among newly opened accounts'
'How can I help you query your database?'"""
                return {
                    "success": False,
                    "error": error_message
                }

            # 1. Check Cache (skip if feedback is provided since it's a correction)
            if not feedback:
                query_hash = hashlib.sha256(natural_language_query.encode()).hexdigest()
                cached_sql = get_cached_query(self.db_manager, query_hash)
                
                if cached_sql:
                    logger.info("Cache hit! Returning cached SQL.")
                    return {
                        "success": True,
                        "sql_query": cached_sql,
                        "cached": True
                    }

            # 2. Get last SQL from memory (for modification detection)
            last_sql = None
            if self.memory_manager:
                last_sql = self.memory_manager.get_last_sql()
            
            # 3. Classify query type (NEW_QUERY or MODIFICATION)
            query_type = self._classify_query_type(natural_language_query, last_sql)
            logger.info(f"Query classified as: {query_type}")

            # 4. Prepare enhanced context with memory and feedback
            enhanced_context = self.prepare_enhanced_context(
                query=natural_language_query,
                feedback=feedback,
                use_enhanced_ddl=True
            )
            
            # 5. Create context strings
            conversation_history = enhanced_context["conversation_history"]
            schema_context = enhanced_context["schema_with_comments"]
            user_feedback = enhanced_context["user_feedback"]
            
            # Create tasks with enhanced context
            analysis_task = Task(
                description=f"""Analyze the following natural language query and identify:
                1. The main entities/tables involved
                2. The type of operation (SELECT, INSERT, UPDATE, DELETE)
                3. Any filtering conditions (WHERE, HAVING)
                4. Any aggregation requirements (COUNT, SUM, AVG, etc.)
                5. Any sorting requirements
                
                ## Conversation History:
                {conversation_history}
                
                ## Current Query: 
                {natural_language_query}
                
                ## Database Schema Context (with column comments):
                {schema_context}
                
                ## Previous Feedback:
                {user_feedback}
                
                Provide a detailed analysis in JSON format:
                {{
                    "entities": ["list of main tables"],
                    "operation": "SELECT/INSERT/UPDATE/DELETE",
                    "filters": ["list of filtering conditions"],
                    "aggregations": ["list of aggregation functions needed"],
                    "sorting": ["list of sorting requirements"],
                    "feedback_notes": "any corrections needed based on feedback"
                }}""",
                agent=self.agents["sql_analyst"],
                expected_output="JSON analysis of the query requirements",
                callback=self._log_task_output
            )
            
            schema_task = Task(
                description=f"""Based on the analysis, provide detailed database context including:
                1. Table relationships and foreign keys based on DDLs
                2. Data types and constraints
                3. Column descriptions and their meanings
                4. Indexing considerations
                
                ## Schema Context (DDLs with column comments):
                {schema_context}
                
                ## Previous Feedback/Corrections:
                {user_feedback}
                
                Provide database-specific insights for SQL generation.
                Pay special attention to:
                - Column names and descriptions (look for comments after --)
                - Any corrections mentioned in feedback
                - Proper JOIN conditions""",
                agent=self.agents["db_expert"],
                expected_output="Database context and insights",
                callback=self._log_task_output
            )
            
            # Create generation task - adapt based on query type
            if query_type == 'MODIFICATION' and last_sql:
                # Modification mode: provide last SQL and ask for targeted changes
                generation_task = Task(
                    description=f"""MODIFICATION REQUEST: The user wants to modify the previous SQL query.
                    
                    ## Last SQL Generated:
                    {last_sql}
                    
                    ## User's Modification Request:
                    {natural_language_query}
                    
                    ## Conversation History:
                    {conversation_history}
                    
                    ## Schema Context (with column descriptions after --):
                    {schema_context}
                    
                    ## Previous Feedback/Corrections:
                    {user_feedback}
                    
                    CRITICAL INSTRUCTIONS FOR MODIFICATION:
                    - START with the last SQL query shown above
                    - Apply ONLY the specific changes requested by the user
                    - Keep all other parts of the query unchanged
                    - If user says "add column X", add it to SELECT clause
                    - If user says "fix join", correct the JOIN condition only
                    - If user says "filter by Y", add/modify WHERE clause
                    - Output ONLY the modified raw SQL query
                    - NO "Thought:", "Final Answer:", or explanations
                    - NO markdown formatting (no ```sql)
                    - Start directly with SELECT/INSERT/UPDATE/DELETE
                    """,
                    agent=self.agents["sql_developer"],
                    expected_output="Modified SQL query based on last SQL",
                    callback=self._log_task_output
                )
            else:
                # New query mode: standard generation
                generation_task = Task(
                    description=f"""Generate a valid PostgreSQL SQL query for the following request.
                    
                    ## Current User Query:
                    {natural_language_query}
                    
                    ## Conversation History:
                    {conversation_history}
                    
                    ## Schema Context (with column descriptions after --):
                    {schema_context}
                    
                    ## Previous Feedback/Corrections:
                    {user_feedback}
                    
                    CRITICAL INSTRUCTIONS:
                    - Output ONLY the raw SQL query
                    - NO "Thought:", "Final Answer:", or explanations
                    - NO markdown formatting (no ```sql)
                - Start the output directly with the SQL verb (SELECT, INSERT, etc.)
                - Do not include "I now can give a great answer"
                - If feedback mentions corrections (e.g., "use customer_no instead of customer_id"), apply them
                - Pay attention to column names in the schema comments
                """,
                agent=self.agents["sql_developer"],
                expected_output="Raw SQL query string only",
                callback=self._log_task_output
            )
            
            # Create crew
            crew = Crew(
                agents=list(self.agents.values()),
                tasks=[analysis_task, schema_task, generation_task],
                process=Process.sequential,
                verbose=True
            )
            
            # Execute crew
            result = crew.kickoff()
            sql_str = getattr(result, "raw", None)
            logging.info(f"Final result raw: {sql_str}")

            if sql_str is None:
                # If raw is None, try to use the result directly if it's a string
                if isinstance(result, str):
                    sql_str = result
                else:
                    logger.error("crew.kickoff() did not return string raw.")
                    return {
                        "success": False,
                        "error": "Failed to generate SQL from agent output"
                    }

            # Extract SQL from result
            sql_query = self._extract_sql_from_result(str(sql_str))
            
            if not sql_query:
                # Fallback: if extraction failed, try to return the raw string if it looks like SQL
                raw_str = str(sql_str).strip()
                if re.search(r'(?i)^(SELECT|INSERT|UPDATE|DELETE|WITH)', raw_str):
                     sql_query = raw_str
                else:
                    return {
                        "success": False,
                        "error": "Failed to extract valid SQL from agent output"
                    }
            
            # Cache the successful result (only if no feedback was used)
            if not feedback:
                cache_query(self.db_manager, query_hash, natural_language_query, sql_query)
            
            # Save to memory if memory manager is available
            if self.memory_manager:
                try:
                    self.memory_manager.add_interaction(
                        user_query=natural_language_query,
                        generated_sql=sql_query,
                        feedback=feedback,
                        feedback_type='correction' if feedback else None
                    )
                except Exception as e:
                    logger.warning(f"Failed to save interaction to memory: {str(e)}")
            
            return {
                "success": True,
                "sql_query": sql_query,
                "cached": False
            }
            
        except Exception as e:
            logger.error(f"Error generating SQL: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }

    def _log_task_output(self, task_output):
        """Callback to log task output."""
        agent_name = "Unknown Agent"
        if hasattr(task_output, "agent"):
            agent_name = task_output.agent
        
        # Extract the actual output content
        output_content = task_output
        if hasattr(task_output, "raw"):
             output_content = task_output.raw
             
        logger.info(f"[{agent_name}] Task Output: {output_content}")

    def _extract_sql_from_result(self, result: str) -> Optional[str]:
        """
        Try to extract the first valid SQL DML query (SELECT/INSERT/UPDATE/DELETE)
        from SQL code blocks or inline text.
        """
        # 1. Try to find SQL queries in markdown code blocks first
        code_blocks = re.findall(r"```sql(.*?)```", result, flags=re.DOTALL | re.IGNORECASE)
        if not code_blocks:
             code_blocks = re.findall(r"```(.*?)```", result, flags=re.DOTALL | re.IGNORECASE)
             
        for block in code_blocks:
            clean_block = block.strip()
            # Basic validation to ensure it looks like SQL
            if re.search(r'(?i)^(SELECT|INSERT|UPDATE|DELETE|WITH)', clean_block):
                return clean_block
            
        # 2. Fallback: try to find inline SQL query in the main text
        # Look for a pattern that starts with a SQL keyword and ends with a semicolon
        # We relax the regex to capture multi-line queries more reliably
        match = re.search(
            r'(?i)(SELECT|INSERT|UPDATE|DELETE|WITH)\s+.*?;',
            result,
            flags=re.DOTALL
        )
        if match:
            return match.group(0).strip()
            
        return None
    
    def execute_sql(self, sql_query: str) -> Dict[str, Any]:
        """Execute SQL query and return results."""
        try:
            # Clean comments before execution
            clean_query = re.sub(r'--.*', '', sql_query)
            # Remove empty lines
            clean_query = "\n".join([line for line in clean_query.split('\n') if line.strip()])
            
            # Only execute the first statement if multiple are present (safety)
            if ";" in clean_query:
                clean_query = clean_query.split(";")[0]
            
            df = self.db_manager.execute_query(clean_query)
            return {
                "success": True,
                "data": df.to_dict('records'),
                "columns": df.columns.tolist(),
                "row_count": len(df)
            }
        except Exception as e:
            logger.error(f"Error executing SQL: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def close(self):
        """Close database connection."""
        self.db_manager.close()

if __name__ == "__main__":
    sql_agent = SQLAgent()
    sql_agent._create_agents()
    print(sql_agent.agents)

import json
import logging
from typing import Dict, Any, List, Optional, Tuple
from crewai import Agent, Task, Crew, Process
from src.database_manager import DatabaseManager
from src.ollama_llm import OllamaManager, OllamaLLM

logger = logging.getLogger(__name__)

class SQLAgent:
    """Main SQL Agent class using CrewAI framework."""
    
    def __init__(self, config_path: str = "config/database_config.json"):
        """Initialize SQL Agent with all components."""
        self.config_path = config_path
        self.db_manager = DatabaseManager(config_path)
        self.ollama_manager = OllamaManager(config_path)
        self.llm = self.ollama_manager.llm
        
        # Load sample queries for context
        self.sample_queries = self._load_sample_queries()
        
        # Initialize agents
        self.agents = self._create_agents()
        
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

        # self.llm = OllamaLLM(model="llama3.2:latest", base_url="http://localhost:11434")
        self.llm = OllamaLLM(provider="ollama", model="llama3.2:latest", base_url="http://localhost:11434")
        print(self.llm)
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
        
        # Query Validator Agent
        query_validator = Agent(
            role="Query Validator",
            goal="Validate SQL queries for correctness and optimization",
            backstory="""You are a SQL validation expert who ensures queries are 
            syntactically correct, follow best practices, and are optimized for 
            the specific database schema.""",
            verbose=True,
            allow_delegation=False,
            llm=self.llm
        )
        
        return {
            "sql_analyst": sql_analyst,
            "db_expert": db_expert,
            "sql_developer": sql_developer,
            "query_validator": query_validator
        }
    
    def connect_database(self) -> bool:
        """Connect to the database."""
        return self.db_manager.connect()
    
    def test_ollama_connection(self) -> bool:
        """Test connection to Ollama service."""
        return self.ollama_manager.test_connection()
    
    def _create_schema_context(self, relevant_tables: List[str]) -> str:
        """Create schema context for the given tables."""
        try:
            schema = self.db_manager.get_database_schema()
            context_parts = []
            
            for table_name in relevant_tables:
                if table_name in schema["tables"]:
                    table_info = schema["tables"][table_name]
                    
                    # Table header
                    context_parts.append(f"Table: {table_name}")
                    
                    # Columns
                    columns = []
                    for col in table_info["columns"]:
                        nullable = "NULL" if col["nullable"] else "NOT NULL"
                        default = f" DEFAULT {col['default']}" if col["default"] else ""
                        columns.append(f"  - {col['name']}: {col['type']} {nullable}{default}")
                    
                    context_parts.append("Columns:")
                    context_parts.extend(columns)
                    
                    # Primary keys
                    if table_info["primary_keys"]:
                        context_parts.append(f"Primary Keys: {', '.join(table_info['primary_keys'])}")
                    
                    # Foreign keys
                    if table_info["foreign_keys"]:
                        fk_info = []
                        for fk in table_info["foreign_keys"]:
                            fk_info.append(f"{', '.join(fk['constrained_columns'])} -> {fk['referred_table']}.{', '.join(fk['referred_columns'])}")
                        context_parts.append(f"Foreign Keys: {'; '.join(fk_info)}")
                    
                    context_parts.append("")  # Empty line for separation
            
            return "\n".join(context_parts)
            
        except Exception as e:
            logger.error(f"Error creating schema context: {str(e)}")
            return f"Error: Could not retrieve schema information - {str(e)}"
    
    def _create_examples_context(self) -> str:
        """Create examples context from sample queries."""
        if not self.sample_queries:
            return "No example queries available."
        
        examples = []
        for i, query_info in enumerate(self.sample_queries[:5], 1):  # Limit to 5 examples
            examples.append(f"Example {i}:")
            examples.append(f"  Natural Language: {query_info['natural_language']}")
            examples.append(f"  SQL: {query_info['sql']}")
            examples.append("")
        
        return "\n".join(examples)
    
    def generate_sql(self, natural_language_query: str) -> Dict[str, Any]:
        """Generate SQL query using CrewAI agents."""
        
        try:
            # Get relevant tables
            relevant_tables = self.db_manager.get_relevant_tables(natural_language_query)
            
            # Create context
            schema_context = self._create_schema_context(relevant_tables)
            examples_context = self._create_examples_context()
            
            # Create tasks
            analysis_task = Task(
                description=f"""Analyze the following natural language query and identify:
                1. The main entities/tables involved
                2. The type of operation (SELECT, INSERT, UPDATE, DELETE)
                3. Any filtering conditions
                4. Any aggregation requirements
                5. Any sorting requirements
                
                Query: {natural_language_query}
                
                Database Schema Context:
                {schema_context}
                
                Provide a detailed analysis in JSON format:
                {{
                    "entities": ["list of main tables"],
                    "operation": "SELECT/INSERT/UPDATE/DELETE",
                    "filters": ["list of filtering conditions"],
                    "aggregations": ["list of aggregation functions needed"],
                    "sorting": ["list of sorting requirements"],
                    "joins": ["list of required table joins"]
                }}""",
                agent=self.agents["sql_analyst"],
                expected_output="JSON analysis of the query requirements"
            )
            
            schema_task = Task(
                description=f"""Based on the analysis, provide detailed database context including:
                1. Table relationships and foreign keys
                2. Data types and constraints
                3. Sample data patterns
                4. Indexing considerations
                
                Schema Context:
                {schema_context}
                
                Provide database-specific insights for SQL generation.""",
                agent=self.agents["db_expert"],
                expected_output="Database context and insights"
            )
            
            generation_task = Task(
                description=f"""Generate a PostgreSQL SQL query based on the analysis and database context.
                
                Natural Language Query: {natural_language_query}
                
                Example Queries for Reference:
                {examples_context}
                
                Instructions:
                1. Use the analysis to understand requirements
                2. Apply database context for proper table relationships
                3. Generate syntactically correct PostgreSQL SQL
                4. Include proper JOINs, WHERE clauses, and aggregations
                5. Return ONLY the SQL query, no explanations
                
                Generate the SQL query:""",
                agent=self.agents["sql_developer"],
                expected_output="Valid PostgreSQL SQL query"
            )
            
            validation_task = Task(
                description="""Validate the generated SQL query for:
                1. Syntax correctness
                2. Proper table and column references
                3. Efficient structure
                4. Best practices adherence
                
                Provide validation result and any suggestions for improvement.""",
                agent=self.agents["query_validator"],
                expected_output="SQL validation result and suggestions"
            )
            
            # Create crew
            crew = Crew(
                agents=list(self.agents.values()),
                tasks=[analysis_task, schema_task, generation_task, validation_task],
                process=Process.sequential,
                verbose=True
            )
            
            # Execute crew
            result = crew.kickoff()
            
            # Extract SQL from result
            sql_query = self._extract_sql_from_result(result)
            
            return {
                "success": True,
                "sql_query": sql_query,
                "natural_language_query": natural_language_query,
                "relevant_tables": relevant_tables,
                "crew_result": result,
                "schema_context": schema_context
            }
            
        except Exception as e:
            logger.error(f"Error in SQL generation: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "natural_language_query": natural_language_query
            }
    
    def _extract_sql_from_result(self, result: str) -> str:
        """Extract SQL query from crew result."""
        # Look for SQL patterns in the result
        lines = result.split('\n')
        sql_lines = []
        in_sql = False
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            # Check if line contains SQL keywords
            if any(keyword in line.upper() for keyword in ['SELECT', 'INSERT', 'UPDATE', 'DELETE', 'WITH']):
                in_sql = True
            
            if in_sql:
                sql_lines.append(line)
                
                # Check if line ends with semicolon
                if line.endswith(';'):
                    break
        
        if sql_lines:
            return ' '.join(sql_lines)
        else:
            # Fallback: return the entire result
            return result.strip()
    
    def execute_sql(self, sql_query: str) -> Dict[str, Any]:
        """Execute SQL query and return results."""
        try:
            df = self.db_manager.execute_query(sql_query)
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
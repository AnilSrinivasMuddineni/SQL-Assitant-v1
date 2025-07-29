import json
import requests
from typing import Dict, Any, List, Optional
import logging
from langchain.llms.base import LLM
# from langchain.callbacks.manager import CallbackManagerForLLM
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

class OllamaLLM(LLM, BaseModel):
    """Custom LLM class for Ollama integration."""
    
    base_url: str = Field(default="http://localhost:11434")
    model: str = Field(default="tinyllama:1.1b-chat")
    temperature: float = Field(default=0.7)
    max_tokens: int = Field(default=2048)
    
    @property
    def _llm_type(self) -> str:
        return "ollama"
    
    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager = None,
    ) -> str:
        """Generate response from Ollama model."""
        try:
            response = requests.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": self.temperature,
                        "num_predict": self.max_tokens
                    }
                },
                timeout=60
            )
            
            if response.status_code == 200:
                result = response.json()
                return result.get("response", "")
            else:
                logger.error(f"Ollama API error: {response.status_code} - {response.text}")
                return "Error: Unable to generate response from Ollama model."
                
        except requests.exceptions.RequestException as e:
            logger.error(f"Request error: {str(e)}")
            return f"Error: Failed to connect to Ollama at {self.base_url}"
        except Exception as e:
            logger.error(f"Unexpected error: {str(e)}")
            return f"Error: {str(e)}"
    
    @property
    def _identifying_params(self) -> Dict[str, Any]:
        """Get identifying parameters."""
        return {
            "base_url": self.base_url,
            "model": self.model,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens
        }

class OllamaManager:
    """Manager class for Ollama operations."""
    
    def __init__(self, config_path: str = "config/database_config.json"):
        """Initialize Ollama manager with configuration."""
        self.config = self._load_config(config_path)
        self.base_url = self.config['ollama']['base_url']
        self.model = self.config['ollama']['model']
        self.llm = OllamaLLM(
            base_url=self.base_url,
            model=self.model
        )
    
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from JSON file."""
        try:
            with open(config_path, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            logger.error(f"Configuration file not found: {config_path}")
            raise
        except json.JSONDecodeError:
            logger.error(f"Invalid JSON in configuration file: {config_path}")
            raise
    
    def test_connection(self) -> bool:
        """Test connection to Ollama service."""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=10)
            if response.status_code == 200:
                models = response.json().get("models", [])
                model_names = [model.get("name", "") for model in models]
                
                if self.model in model_names:
                    logger.info(f"Ollama connection successful. Model {self.model} available.")
                    return True
                else:
                    logger.warning(f"Model {self.model} not found. Available models: {model_names}")
                    return False
            else:
                logger.error(f"Failed to connect to Ollama: {response.status_code}")
                return False
                
        except requests.exceptions.RequestException as e:
            logger.error(f"Connection error: {str(e)}")
            return False
    
    def generate_sql(self, prompt: str, schema_context: str, examples: str = "") -> str:
        """Generate SQL query from natural language prompt."""
        
        system_prompt = f"""You are an expert SQL developer. Your task is to convert natural language queries into valid PostgreSQL SQL statements.

Database Schema Context:
{schema_context}

Example Queries for Reference:
{examples}

Instructions:
1. Analyze the user's natural language query
2. Use the provided database schema to understand table structures and relationships
3. Generate a valid PostgreSQL SQL query
4. Ensure the query is syntactically correct and follows PostgreSQL conventions
5. Use appropriate JOINs when multiple tables are involved
6. Include proper WHERE clauses for filtering
7. Use appropriate aggregation functions when needed
8. Return ONLY the SQL query, no explanations or additional text

User Query: {prompt}

SQL Query:"""

        try:
            response = self.llm._call(system_prompt)
            return response.strip()
        except Exception as e:
            logger.error(f"Error generating SQL: {str(e)}")
            return f"Error: Failed to generate SQL query - {str(e)}"
    
    def validate_sql(self, sql_query: str) -> Dict[str, Any]:
        """Validate generated SQL query."""
        validation_prompt = f"""Validate the following PostgreSQL SQL query. Check for:
1. Syntax errors
2. Missing semicolon
3. Proper table and column names
4. Valid SQL structure

Query: {sql_query}

Provide validation result in JSON format:
{{
    "is_valid": true/false,
    "errors": ["list of errors if any"],
    "suggestions": ["list of suggestions if any"]
}}"""

        try:
            response = self.llm._call(validation_prompt)
            # Try to parse JSON response
            try:
                import json
                return json.loads(response)
            except json.JSONDecodeError:
                return {
                    "is_valid": True,
                    "errors": [],
                    "suggestions": ["Could not parse validation response"]
                }
        except Exception as e:
            logger.error(f"Error validating SQL: {str(e)}")
            return {
                "is_valid": False,
                "errors": [f"Validation failed: {str(e)}"],
                "suggestions": []
            } 
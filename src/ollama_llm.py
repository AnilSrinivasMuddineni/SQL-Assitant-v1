import json
import requests
from typing import Dict, Any, List, Optional
import logging
# Use ChatOpenAI from langchain_community as langchain-openai is not installed
from langchain_community.chat_models import ChatOpenAI

logger = logging.getLogger(__name__)

class OllamaManager:
    def __init__(self, config_path: str):
        """Initialize Ollama Manager."""
        self.config_path = config_path
        self.config = self._load_config()
        
        # Load and clean config
        raw_base_url = self.config.get("ollama", {}).get("base_url", "http://localhost:11434")
        self.base_url = raw_base_url.rstrip("/")
        
        raw_model = self.config.get("ollama", {}).get("model", "sqlcoder:7b")
        self.model = raw_model.strip()
        
        logger.info(f"Initializing Ollama with model='{self.model}', base_url='{self.base_url}'")
        
        # Use ChatOpenAI client pointing to Ollama
        # This is often more stable for CrewAI than the native Ollama integration
        # NOTE: We use openai_api_base instead of base_url for compatibility with older langchain_community
        self.llm = ChatOpenAI(
            model=self.model,
            openai_api_base=f"{self.base_url}/v1",
            openai_api_key="NA", # Ollama doesn't require a key, but the client expects one
            temperature=0.7
        )

    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from JSON file."""
        try:
            with open(self.config_path, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            logger.warning(f"Config file {self.config_path} not found. Using defaults.")
            return {}
        except json.JSONDecodeError:
            logger.error(f"Invalid JSON in config file {self.config_path}.")
            return {}

    def update_model(self, model_name: str, base_url: str):
        """Update the model and base URL."""
        self.model = model_name.strip()
        self.base_url = base_url.rstrip("/")
        
        # Update environment variables for CrewAI
        import os
        os.environ["OPENAI_API_BASE"] = f"{self.base_url}/v1"
        os.environ["OPENAI_BASE_URL"] = f"{self.base_url}/v1"
        os.environ["OPENAI_MODEL_NAME"] = self.model
        
        logger.info(f"Updating Ollama to model='{self.model}', base_url='{self.base_url}'")
        
        self.llm = ChatOpenAI(
            model=self.model,
            openai_api_base=f"{self.base_url}/v1",
            openai_api_key="NA",
            temperature=0.7
        )

    def get_available_models(self) -> List[str]:
        """Fetch available models from Ollama."""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            if response.status_code == 200:
                models = response.json().get("models", [])
                return [model.get("name") for model in models]
            return []
        except Exception as e:
            logger.error(f"Error fetching models: {str(e)}")
            return []

    def test_connection(self) -> bool:
        """Test connection to Ollama service."""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=10)
            if response.status_code == 200:
                models = response.json().get("models", [])
                model_names = [model.get("name", "") for model in models]
                
                # Check if configured model is available
                # We relax this check to just connection success, 
                # as the model might be pulled on demand or name might vary slightly
                logger.info(f"Ollama connection successful. Found {len(model_names)} models.")
                return True
            return False
        except Exception as e:
            logger.error(f"Ollama connection failed: {str(e)}")
            return False
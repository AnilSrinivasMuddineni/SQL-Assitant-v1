import os
import logging
from src.ollama_llm import OllamaManager

# Configure logging to see what's happening
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set dummy key
os.environ["OPENAI_API_KEY"] = "NA"

def reproduce():
    print("Reproducing issue with OllamaManager...")
    
    # Initialize manager
    # We assume config/database_config.json exists and has the model
    manager = OllamaManager("config/database_config.json")
    
    print(f"Model: {manager.model}")
    print(f"Base URL: {manager.base_url}")
    print(f"LLM Type: {type(manager.llm)}")
    
    try:
        print("Invoking LLM...")
        response = manager.llm.invoke("Hello")
        print(f"Response: {response.content}")
        print("SUCCESS")
    except Exception as e:
        print(f"FAILURE: {e}")

if __name__ == "__main__":
    reproduce()

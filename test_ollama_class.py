from langchain_community.llms import Ollama
import os

# Set env vars just in case
os.environ["OPENAI_API_KEY"] = "NA"
os.environ["OPENAI_API_BASE"] = "http://localhost:11434/v1"

def test_ollama_completion():
    print("Testing langchain_community.llms.Ollama (Completion API)...")
    try:
        llm = Ollama(model="sqlcoder:7b", base_url="http://localhost:11434")
        response = llm.invoke("SELECT * FROM users")
        print(f"Response: {response}")
        print("SUCCESS")
    except Exception as e:
        print(f"FAILURE: {e}")

if __name__ == "__main__":
    test_ollama_completion()

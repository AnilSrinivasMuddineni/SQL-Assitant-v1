import os
from langchain_community.chat_models import ChatOpenAI

# Set dummy key
os.environ["OPENAI_API_KEY"] = "NA"

def test_params():
    print("Testing ChatOpenAI parameters...")
    
    # Test 1: Using base_url (Current approach)
    print("\n1. Testing with base_url='http://localhost:11434/v1'...")
    try:
        llm = ChatOpenAI(
            model="sqlcoder:7b",
            base_url="http://localhost:11434/v1",
            api_key="NA",
            temperature=0.7
        )
        # We just want to see where it tries to connect. 
        # If it hits OpenAI, it will fail with 401 or 404 (if model not found there).
        # If it hits localhost, it might work or fail with connection error if ollama is down.
        llm.invoke("Hello")
        print("   SUCCESS: base_url worked.")
    except Exception as e:
        print(f"   FAILURE: {e}")

    # Test 2: Using openai_api_base (Legacy approach)
    print("\n2. Testing with openai_api_base='http://localhost:11434/v1'...")
    try:
        llm = ChatOpenAI(
            model="sqlcoder:7b",
            openai_api_base="http://localhost:11434/v1",
            openai_api_key="NA",
            temperature=0.7
        )
        llm.invoke("Hello")
        print("   SUCCESS: openai_api_base worked.")
    except Exception as e:
        print(f"   FAILURE: {e}")

if __name__ == "__main__":
    test_params()

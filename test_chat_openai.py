import os
from langchain_community.chat_models import ChatOpenAI

# Set dummy key
os.environ["OPENAI_API_KEY"] = "NA"

def test_chat():
    print("Testing ChatOpenAI with llama3.1:8b...")
    try:
        llm = ChatOpenAI(
            model="llama3.1:8b",
            base_url="http://localhost:11434/v1",
            api_key="NA",
            temperature=0.7
        )
        
        print("Invoking LLM...")
        response = llm.invoke("Hello, are you there?")
        print(f"Response: {response.content}")
        print("SUCCESS")
        
    except Exception as e:
        print(f"FAILURE: {e}")

if __name__ == "__main__":
    test_chat()

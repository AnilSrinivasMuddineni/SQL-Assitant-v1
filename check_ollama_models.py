import requests
import json

def check_models():
    base_url = "http://localhost:11434"
    try:
        print(f"Querying {base_url}/api/tags ...")
        response = requests.get(f"{base_url}/api/tags", timeout=5)
        if response.status_code == 200:
            data = response.json()
            models = data.get("models", [])
            print(f"Found {len(models)} models:")
            for model in models:
                print(f" - Name: '{model.get('name')}'")
                print(f"   Model: '{model.get('model')}'")
                print(f"   Digest: {model.get('digest')[:12]}...")
                print("---")
        else:
            print(f"Error: Status code {response.status_code}")
            print(response.text)
    except Exception as e:
        print(f"Exception: {e}")

if __name__ == "__main__":
    check_models()

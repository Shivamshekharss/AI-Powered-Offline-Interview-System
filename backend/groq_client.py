# backend/groq_client.py

import requests
from backend.config.settings import get_settings

settings = get_settings()

def groq_chat(messages, temperature: float = None):
    """
    Send chat completion to Groq (OpenAI compatible).
    """
    api_key = settings.groq_api_key
    model = settings.groq_model
    temperature = temperature if temperature is not None else settings.groq_temperature

    url = f"{settings.groq_api_url}/chat/completions"

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
    }

    response = requests.post(url, headers=headers, json=payload)
    if response.status_code != 200:
        print("STATUS:", response.status_code)
        print("RESPONSE:", response.text)
        raise SystemExit()


    data = response.json()

    return data["choices"][0]["message"]["content"]


if __name__ == "__main__":
    print("🔄 Sending test request to Groq...\n")
    try:
        reply = groq_chat([
            {"role": "system", "content": "You are a friendly assistant"},
            {"role": "user", "content": "Say hello!"}
        ])
        print("✅ Groq Response:", reply)
    except Exception as e:
        print("\n❌ Error occurred:")
        print(str(e))

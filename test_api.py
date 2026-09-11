"""Quick API connectivity test — run this first to verify your Groq key works."""
import os
from dotenv import load_dotenv
load_dotenv()
from groq import Groq

api_key = os.getenv("GROQ_API_KEY")
print(f"API Key found: {'YES' if api_key else 'NO'}")
if api_key:
    print(f"Key starts with: {api_key[:8]}...")

try:
    client = Groq(api_key=api_key)
    response = client.chat.completions.create(
        messages=[{"role": "user", "content": "Reply with just the word: working"}],
        model="qwen/qwen3.8-27b",
        temperature=0.0
    )
    print(f"API Response: {response.choices[0].message.content}")
    print("✅ Groq API is working correctly!")
except Exception as e:
    print(f"❌ API Error: {type(e).__name__}: {e}")

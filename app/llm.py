import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY")
client_llm = OpenAI(api_key=openai_api_key)

def get_llm_response(prompt: str, model: str = "gpt-4o-mini") -> str:
    """Centralized function to get responses from OpenAI."""
    response = client_llm.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content

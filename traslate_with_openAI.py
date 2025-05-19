import openai
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

# Read the API key from the .env file
openai.api_key = os.getenv("OPENAI_API_KEY")

def translate_with_gpt(text):
    messages = [
        {"role": "system", "content": "Translate from Japanese to natural English."},
        {"role": "user", "content": text}
    ]
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=messages
    )
    return response.choices[0].message["content"]

translated = translate_with_gpt("こんにちは、皆さん")
print(translated)
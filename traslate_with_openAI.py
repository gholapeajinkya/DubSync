import openai
from dotenv import load_dotenv
import os
import json
from openai import AzureOpenAI
# Load environment variables from .env file
load_dotenv()

# Read the API key from the .env file
api_type = "azure"
api_key = os.getenv("OPENAI_API_KEY")
api_base = os.getenv("OPENAI_API_BASE")
api_version = "2025-01-01-preview"  # or latest supported version
DEPLOYMENT_NAME = "gpt-4-dubbing"  # Your Azure OpenAI deployment name

# Create a client using Azure credentials
client = AzureOpenAI(
    api_key=api_key,
    api_version=api_version,
    azure_endpoint=api_base,
)


def translate_with_gpt(segments):
    # segment_text = ""
    # for idx, segment in enumerate(segments):
    #     segment_text += f"\nSegment {idx + 1}:\nJapanese: {segment['text']}\nTranslation: {segment['translation']}\n"
    prompt = f"""
        You are a dubbing script writer. Rewrite the following translation into a natural, expressive line suitable for English 
        anime dubbing.
        It should capture the tone of the original Japanese line, and based on previous conversation make sense of the next
        line and be spoken within words.
        This is segment object:
        {segments} contains raw text Japanese and English translation iterate over it.
        Return the rewritten line in same format."""

    response = client.chat.completions.create(
        model=DEPLOYMENT_NAME,
        messages=[
            {"role": "system",
                "content": "You are a professional anime dubbing scriptwriter."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.7
    )
    return response.choices[0].message.content


# load a json file
with open('translation_segments.json', 'r') as f:
    data = json.load(f)
    translated = translate_with_gpt(data)
    print(translated)
    # Save the translated content into a file
    output_file = 'translated_output.py'
    with open(output_file, 'w') as f:
        f.write(translated)

    print(f"Translated content has been saved to {output_file}")

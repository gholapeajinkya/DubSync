from state import AgentState
import os
from langchain_openai import AzureChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
import ast

# Azure OpenAI configuration
llm = AzureChatOpenAI(
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION"),    
    azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
)

def clean_response_text(text):
    text = text.strip()
    if text.startswith("```json") and text.endswith("```"):
        text = "\n".join(text.strip("`").split("\n")[1:])
    # Handle both straight and curly apostrophes
    unwanted_prefixes = [
        "Here's the rewritten script for the dubbing:",
        "Here's the rewritten script for the dubbing:",  # curly apostrophe
    ]
    for prefix in unwanted_prefixes:
        if text.startswith(prefix):
            text = text[len(prefix):].strip()
            break
    # Also try to find the JSON array start
    if not text.startswith("["):
        bracket_pos = text.find("[")
        if bracket_pos != -1:
            text = text[bracket_pos:]
    return text

def translation_node(state: AgentState) -> AgentState:
    """
    Translates the transcribed text segments into the target language (e.g., English) and saves the translations to the state for further processing.
    """
    transcription_segments = state.get("transcription_segments")
    input_segments = []
    input_language = "Japanese"  # Assuming input is Japanese, adjust as needed
    output_language = "English"  # Target language for dubbing
    for segment in transcription_segments:
        input_segments.append({
            "id": segment["id"],
            "text": segment["text"],
            "start": segment["start"],
            "end": segment["end"]
        })
    prompt = f"""
        You are an expert anime, movie, series dubbing scriptwriter.
        Your job is to take a list of {input_segments} from a {input_language.lower()} video and rewrite the {output_language} translations to sound natural, expressive, and emotionally aligned with how the lines would be spoken in an English dub.

        Each segment contains:
        - `start` and `end` timestamps
        - `id`: The segment ID
        - `text`: The original {input_language.lower()} line

        Your goal:
        - Translate the {input_language.lower()} line into {output_language.lower()}.
        - Make small natural rewrites to the {output_language.lower()} translation.
        - Add filler sounds like "uh", "hmm", "ahh", or stuttering where it fits the character's tone.
        - Preserve emotional nuance (anger, sarcasm, nervousness, etc).
        - Keep the rewritten line short enough to be spoken within the original segment’s timing (`end`-`start`).

        Always consider **previous lines** and **what comes next**, and ensure the dialogue flows naturally across segments.

        Return the rewritten translation line and id in json format, and make sure to start the respond with 'Here's the rewritten script for the dubbing:'
        """

    messages = [
        SystemMessage(content="You are a professional anime, movie, series dubbing scriptwriter."),
        HumanMessage(content=prompt)
    ]
    response = llm.invoke(messages)
    response_segment = clean_response_text(response.content.strip())
    response_segment = clean_response_text(response_segment)
    print(f"translate_with_gpt response => \n{response_segment}")
    ai_segments = ast.literal_eval(str(response_segment))
    # Map ai_segments by id for quick lookup
    ai_segments_dict = {s["id"]: s for s in ai_segments}
    # Add translated text into segments
    for segment in transcription_segments:
        ai_seg = ai_segments_dict.get(segment["id"])
        if ai_seg and "translation" in ai_seg:
            segment["translation"] = ai_seg["translation"]
        elif ai_seg and "text" in ai_seg:  # fallback if key is 'text'
            segment["translation"] = ai_seg["text"]
    state["translations"] = transcription_segments
    return state
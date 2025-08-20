# us_gen.py
import json
from langchain_core.language_models.chat_models import BaseChatModel

def generate_user_story_from_notes(meeting_notes_content: str, llm: BaseChatModel) -> tuple[str, str]:
    """
    Generates a user story summary and description from meeting notes using a provided LLM client.
    Requests and parses a JSON response for robustness.
    """
    # --- PROMPT REFINEMENT ---
    # We are adding very specific instructions for the 'summary' field to keep it short and in the correct format.
    prompt = f"""
    Based on the following meeting notes, generate a user story summary and a detailed description.

    Meeting Notes:
    {meeting_notes_content}

    Your response must be a single, valid JSON object with no other text or markdown.
    The JSON object must have exactly two keys: "summary" and "description".

    - "summary": This MUST be a concise, single sentence for a Jira ticket title, strictly under 255 characters. It must follow the standard user story format: "As a [type of user], I want [some goal] so that [benefit]".
    - "description": This should be a detailed breakdown of the user story, including acceptance criteria or key points from the notes.

    Example of a valid response:
    {{
      "summary": "As a data analyst, I want a dashboard to track weekly sales so that I can identify trends.",
      "description": "The dashboard should include key metrics such as total revenue, number of units sold, and average order value. It needs to have filters for product category and region.\\n\\nAcceptance Criteria:\\n1. Dashboard updates daily.\\n2. Data is accurate to within 1% of the source system.\\n3. Filters work correctly."
    }}
    """
    try:
        print("Generating user story from notes using refined prompt...")
        response = llm.invoke(prompt)
        response_text = response.content.strip()

        # Clean potential markdown fences
        if response_text.startswith("```json"):
            response_text = response_text[7:]
        if response_text.endswith("```"):
            response_text = response_text[:-3]

        data = json.loads(response_text)
        summary = data.get("summary", "Error: Summary key missing in LLM response.")
        description = data.get("description", "Error: Description key missing in LLM response.")

        # --- SAFETY NET ---
        # This ensures that even if the model ignores the prompt, we never send a summary
        # to Jira that is too long. We truncate it safely.
        if len(summary) > 255:
            print(f"Warning: LLM generated a summary longer than 255 characters. Truncating.")
            summary = summary[:252] + "..."

        return summary, description
        
    except json.JSONDecodeError as e:
        print(f"JSON Decode Error in us_gen: {e}")
        return "Error: Invalid JSON from LLM", f"The LLM returned a malformed JSON object. Raw response: {response.content}"
    except Exception as e:
        print(f"Failed to generate user story with LLM: {e}")
        return "Error: Story Generation Failed", f"An unexpected error occurred: {e}"
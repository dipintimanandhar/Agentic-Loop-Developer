# adk_developer_agent/tools/developer_tools.py
from langchain.tools import tool
from vertexai.generative_models import GenerativeModel
import os
import sys

# Add the root project directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from config_manager import read_all_config
# Import the new, powerful logic functions
from adk_developer_agent.developer_logic import process_single_python_story, process_single_sql_story

# --- Global Model Instance ---
model_instance = None

def get_config_and_model():
    """Helper to initialize and return config and the core GenerativeModel."""
    global model_instance
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    config_path = os.path.join(project_root, 'config.ini')
    config = read_all_config(config_path)

    if model_instance is None:
        import vertexai
        print("Initializing Vertex AI GenerativeModel for developer tools...")
        vertexai.init(project=config['project_id'], location=config['vertex_ai_location'])
        model_instance = GenerativeModel(config['vertex_ai_model_name'])
        print("GenerativeModel initialized.")
    return config, model_instance

@tool
def process_sql_conversion_story(issue_key: str) -> str:
    """
    Fully processes a Jira story for Oracle to PostgreSQL conversion given a Jira 'issue_key'.
    This tool downloads attachments, runs the conversion and execution logic with an LLM-powered
    feedback loop, and uploads the resulting SQL and execution report back to the Jira story.
    """
    print(f"--- Agent activating REAL SQL Developer Tool for {issue_key} ---")
    try:
        config, model = get_config_and_model()
        # Call the REAL logic function
        result_message = process_single_sql_story(issue_key, config, model)
        return result_message
    except Exception as e:
        # Catch errors from the logic function to report them back to the agent
        return f"A critical error occurred while processing SQL story {issue_key}: {e}"

@tool
def process_python_development_story(issue_key: str) -> str:
    """
    Fully processes a Jira story that requires a Python solution, given a Jira 'issue_key'.
    This tool implements the 'Loop Developer' mode to iteratively generate and test Python code
    until all tests pass. The final, working Python code is uploaded as 'solution.py' to the Jira story.
    """
    print(f"--- Agent activating Python Loop Developer for {issue_key} ---")
    try:
        config, model = get_config_and_model()
        result_message = process_single_python_story(issue_key, config, model)
        return result_message
    except Exception as e:
        return f"An error occurred while processing Python story {issue_key}: {e}"
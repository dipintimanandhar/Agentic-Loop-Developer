# adk_developer_agent/tools/jira_tools.py
import os
import shutil
from jira import JIRA
from langchain.tools import tool
import sys
from typing import List

# Add the root project directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from us_gen import generate_user_story_from_notes
from config_manager import read_all_config

# --- Global Singleton Instances ---
jira_instance = None
jira_project_key = None
llm_instance = None # To hold the initialized LLM for story generation

def get_config():
    """Helper to get the project config robustly."""
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    config_path = os.path.join(project_root, 'config.ini')
    return read_all_config(config_path)

def get_jira_client() -> JIRA:
    """Initializes and returns a Jira client instance using a singleton pattern."""
    global jira_instance, jira_project_key
    if jira_instance is None:
        config = get_config()
        if config and config.get('jira_server'):
            try:
                print("Initializing Jira client...")
                jira_instance = JIRA(
                    server=config['jira_server'],
                    basic_auth=(config['jira_email'], config['jira_api_token'])
                )
                jira_project_key = config['jira_project_key']
                print("Jira client initialized successfully.")
            except Exception as e:
                print(f"Failed to initialize Jira client: {e}")
                raise e # Re-raise exception to be caught by the tool
    return jira_instance

def get_llm_client():
    """Initializes and returns a LangChain LLM client using a singleton pattern."""
    global llm_instance
    if llm_instance is None:
        try:
            from langchain_google_vertexai import ChatVertexAI
            import vertexai
            config = get_config()
            print("Initializing Vertex AI LLM client...")
            vertexai.init(project=config['project_id'], location=config['vertex_ai_location'])
            llm_instance = ChatVertexAI(
                model_name=config['vertex_ai_model_name'],
                temperature=0.1,
                convert_system_message_to_human=True
            )
            print("Vertex AI LLM client initialized successfully.")
        except Exception as e:
            print(f"Failed to initialize LLM client: {e}")
            raise e
    return llm_instance


@tool
def list_jira_stories() -> str:
    """
    Fetches and lists all user stories from the configured Jira project.
    Returns a formatted string of the stories found.
    """
    try:
        jira = get_jira_client()
        jql_query = f'project = "{jira_project_key}" AND issueType = "Story" ORDER BY created DESC'
        issues = jira.search_issues(jql_query, maxResults=50)
        if not issues: return f"No user stories found in project {jira_project_key}."

        result_lines = [f"Found {len(issues)} stories in project {jira_project_key}:"]
        for issue in issues:
            result_lines.append(f"- {issue.key}: {issue.fields.summary} (Status: {issue.fields.status.name})")
        return "\n".join(result_lines)
    except Exception as e:
        return f"Error fetching Jira stories: {e}"

@tool
def process_uploaded_notes_file(uploaded_file_path: str) -> str:
    """
    Processes an uploaded meeting notes file to create a new Jira story.
    The file is saved, its content is used to generate a story summary and description via an LLM,
    and then a new story is created in Jira with the original notes as an attachment.
    The 'uploaded_file_path' is a temporary path to the file provided by the user interface.
    """
    try:
        jira = get_jira_client()
        llm = get_llm_client()

        if not os.path.exists(uploaded_file_path):
            return f"Error: The temporary uploaded file path does not exist: {uploaded_file_path}"

        original_filename = os.path.basename(uploaded_file_path)
        notes_dir = os.path.join(os.path.dirname(__file__), '..', 'mnotes')
        os.makedirs(notes_dir, exist_ok=True)
        permanent_filepath = os.path.join(notes_dir, original_filename)
        shutil.copy(uploaded_file_path, permanent_filepath)

        with open(permanent_filepath, 'r', encoding='utf-8') as f:
            notes_content = f.read()

        summary, description = generate_user_story_from_notes(notes_content, llm)

        if "Error" in summary:
            return f"Failed to generate story content. Reason: {description}"

        print(f"Creating Jira story with summary: {summary}")
        new_story = jira.create_issue(
            project={'key': jira_project_key},
            summary=summary,
            description=description,
            issuetype={'name': 'Story'}
        )

        print(f"Uploading attachment {permanent_filepath} to {new_story.key}")
        with open(permanent_filepath, 'rb') as f_attach:
            jira.add_attachment(issue=new_story.key, attachment=f_attach)

        return f"Successfully created story {new_story.key} from '{original_filename}' and attached the notes."

    except Exception as e:
        return f"An error occurred in process_uploaded_notes_file: {e}"

# --- HELPER FUNCTIONS (Not tools, but used by other tools) ---

def download_attachments(issue_key: str) -> List[str]:
    """Downloads all attachments for an issue and returns their file paths."""
    try:
        jira = get_jira_client()
        issue = jira.issue(issue_key)
        
        issue_download_path = os.path.join('jira_attachments', issue_key)
        os.makedirs(issue_download_path, exist_ok=True)
        
        downloaded_files = []
        for attachment in issue.fields.attachment:
            filepath = os.path.join(issue_download_path, attachment.filename)
            print(f"Downloading '{attachment.filename}' to '{filepath}'...")
            with open(filepath, 'wb') as f:
                f.write(attachment.get())
            downloaded_files.append(filepath)
        return downloaded_files
    except Exception as e:
        print(f"Failed to download attachments for {issue_key}: {e}")
        return []

def upload_attachment(issue_key: str, filepath: str):
    """Uploads a single file to a Jira issue."""
    try:
        jira = get_jira_client()
        print(f"Uploading '{os.path.basename(filepath)}' to issue {issue_key}...")
        with open(filepath, 'rb') as f:
            jira.add_attachment(issue=issue_key, attachment=f)
        print("Upload successful.")
    except Exception as e:
        print(f"Failed to upload attachment to {issue_key}: {e}")
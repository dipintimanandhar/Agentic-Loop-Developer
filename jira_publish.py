from jira import JIRA
import os
from us_gen import generate_user_story_from_notes # Assuming this file exists and is functional

# Jira connection parameters
# JIRA_SERVER = 'https://google-team-i***.atlassian.net' # Replace with your Jira server
# JIRA_EMAIL = '***@google.com' # Replace with your Jira email
# JIRA_API_TOKEN = '' # Replace with your generated API token
# JIRA_PROJECT_KEY = 'PROJ' # Replace with your project key
JIRA_SERVER = 'https://google-team-ic2zy7js.atlassian.net'
JIRA_EMAIL = 'dipintim@google.com'
JIRA_API_TOKEN = 'ATATT3xFfGF0TlTSO2YJOMK2Uw6fl_PCGJxhukAXjtpNc4gkAzK49hPI9flhLY2NqKHwqof-t6pjr-6XJsh5hPJQNcf5qgk7vetn3STI0e-LY2xDVmAaVRr86nsd38nM-Cu_kCbuaoxvPAvUIeJH25g4QICHTpzJC-aO0ODkcm5CMqEWtsxg_YU=E23461DA'
JIRA_PROJECT_KEY = 'KAN'
# Folder containing meeting notes
MEETING_NOTES_FOLDER = 'mnotes'
# Folder to download attachments
ATTACHMENT_DOWNLOAD_FOLDER = 'jira_attachments'

# --- Authentication (Jira) ---
try:
    jira = JIRA(server=JIRA_SERVER, basic_auth=(JIRA_EMAIL, JIRA_API_TOKEN))
    print("Successfully connected to Jira")
except Exception as e:
    print(f"Failed to connect to Jira: {e}")
    exit()

# --- Function to upload an attachment to a Jira issue ---
def upload_attachment_to_story(issue_key, filepath):
    """
    Uploads a file as an attachment to a specified Jira issue.

    :param issue_key: The key of the Jira issue (e.g., 'KAN-123').
    :param filepath: The path to the file to be uploaded.
    """
    if not os.path.exists(filepath):
        print(f"Error: File not found at '{filepath}'. Cannot upload attachment.")
        return

    try:
        issue = jira.issue(issue_key) # Verify issue exists
        print(f"\nUploading attachment '{os.path.basename(filepath)}' to story {issue_key}...")
        with open(filepath, 'rb') as f:
            jira.add_attachment(issue=issue, attachment=f)
        print(f"Successfully uploaded '{os.path.basename(filepath)}' to {issue_key}.")
    except Exception as e:
        print(f"Failed to upload attachment to {issue_key}: {e}")
        if hasattr(e, 'text'):
            print(f"Jira API Error: {e.text}")

# --- Function to download all attachments from a Jira issue ---
def download_attachments_from_story(issue_key):
    """
    Downloads all attachments from a specified Jira issue.

    :param issue_key: The key of the Jira issue (e.g., 'KAN-123').
    """
    try:
        issue = jira.issue(issue_key)
        attachments = issue.fields.attachment

        if not attachments:
            print(f"\nNo attachments found for story {issue_key}.")
            return

        print(f"\nFound {len(attachments)} attachment(s) for story {issue_key}:")
        
        # Create a specific download folder for the issue if it doesn't exist
        issue_download_path = os.path.join(ATTACHMENT_DOWNLOAD_FOLDER, issue_key)
        os.makedirs(issue_download_path, exist_ok=True)
        print(f"Downloading attachments to: '{issue_download_path}'")

        for attachment in attachments:
            filename = attachment.filename
            filepath = os.path.join(issue_download_path, filename)
            print(f"  Downloading '{filename}'...")
            try:
                # Get the raw attachment content
                content = attachment.get()
                with open(filepath, 'wb') as f:
                    f.write(content)
                print(f"  Successfully downloaded '{filename}' to '{filepath}'.")
            except Exception as e_download:
                print(f"  Failed to download '{filename}': {e_download}")

    except Exception as e:
        print(f"Failed to fetch or download attachments for {issue_key}: {e}")
        if hasattr(e, 'text'):
            print(f"Jira API Error: {e.text}")

# --- Main logic to process meeting notes and create Jira story ---
def create_jira_story_from_meeting_notes():
    if not os.path.exists(MEETING_NOTES_FOLDER):
        print(f"Error: Meeting notes folder '{MEETING_NOTES_FOLDER}' not found.")
        return

    meeting_notes_files = [f for f in os.listdir(MEETING_NOTES_FOLDER) if f.endswith('.txt')]
    if not meeting_notes_files:
        print(f"No text files found in '{MEETING_NOTES_FOLDER}'. Please add meeting notes.")
        return

    for filename in meeting_notes_files:
        filepath = os.path.join(MEETING_NOTES_FOLDER, filename)
        print(f"\nProcessing meeting notes from: {filepath}")
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                meeting_notes_content = f.read()
        except Exception as e:
            print(f"Error reading file {filepath}: {e}")
            continue # Skip to the next file

        # Use the imported function from us_gen (ensure us_gen.py is in the same directory or Python path)
        try:
            story_summary, story_description = generate_user_story_from_notes(meeting_notes_content)
        except NameError:
             print("ERROR: 'generate_user_story_from_notes' function is not defined.")
             print("Please ensure 'us_gen.py' exists and the function is correctly imported.")
             return # Stop processing if the core function is missing
        except Exception as e_gen:
            print(f"Error during user story generation for {filename}: {e_gen}")
            story_summary, story_description = None, None


        if story_summary and story_description:
            print(f"Generated Story Summary:\n{story_summary}")
            print(f"Generated Story Description:\n{story_description}")

            ISSUE_TYPE = 'Story' # Or 'User Story', depending on your Jira configuration

            # --- Create the Story in Jira ---
            try:
                new_story = jira.create_issue(
                    project={'key': JIRA_PROJECT_KEY},
                    summary=story_summary,
                    description=story_description,
                    issuetype={'name': ISSUE_TYPE}
                )
                print(f"\nSuccessfully created story: {new_story.key}")
                print(f"View story at: {JIRA_SERVER}/browse/{new_story.key}")

                # --- Upload the meeting notes file as an attachment ---
                upload_attachment_to_story(new_story.key, filepath)

            except Exception as e:
                print(f"Failed to create story for '{filename}': {e}")
                if hasattr(e, 'text'):
                    print(f"Jira API Error: {e.text}")
                elif hasattr(e, 'response') and hasattr(e.response, 'text'):
                    print(f"Jira API Error: {e.response.text}")
        else:
            print(f"Skipping Jira story creation for '{filename}' due to story generation failure or empty content.")

# --- Fetching User Stories (existing functionality) ---
def fetch_existing_user_stories():
    jql_query = f'project = "{JIRA_PROJECT_KEY}" AND issueType = "Story" ORDER BY created DESC'
    issues_list = []
    try:
        issues = jira.search_issues(jql_query, maxResults=100) # `issues` is a ResultList, not a list
        if issues:
            print(f"\nFound {len(issues)} user stories in project {JIRA_PROJECT_KEY}:")
            for issue in issues:
                print(f"- {issue.key}: {issue.fields.summary} (Status: {issue.fields.status.name}, Attachments: {len(issue.fields.attachment)})")
                issues_list.append(issue) # Store the issue object
            return issues_list # Return the list of issue objects
        else:
            print(f"No user stories found in project {JIRA_PROJECT_KEY} with the specified issue types.")
            return []
    except Exception as e:
        print(f"Failed to fetch issues: {e}")
        return []

if __name__ == "__main__":
    # Ensure the 'mnotes' and 'jira_attachments' folders exist
    os.makedirs(MEETING_NOTES_FOLDER, exist_ok=True)
    os.makedirs(ATTACHMENT_DOWNLOAD_FOLDER, exist_ok=True)
    print(f"Ensure your meeting notes are in the '{MEETING_NOTES_FOLDER}' folder.")
    print(f"Attachments will be downloaded to the '{ATTACHMENT_DOWNLOAD_FOLDER}' folder.")
   
    create_jira_story_from_meeting_notes()
    
    print("\n" + "="*50 + "\n") # Separator
    
    fetched_issues = fetch_existing_user_stories() # Renamed to avoid conflict
    
    if fetched_issues:
        print(f"\nFetched {len(fetched_issues)} user stories in project {JIRA_PROJECT_KEY}.")
        # --- Example: Download attachments for the most recently created story if it has any ---
        # The first issue in 'fetched_issues' is the most recent due to 'ORDER BY created DESC'
        most_recent_story = fetched_issues[0]
        if most_recent_story.fields.attachment:
            print(f"\nAttempting to download attachments for the most recent story: {most_recent_story.key}")
            download_attachments_from_story(most_recent_story.key)
        else:
            print(f"\nThe most recent story {most_recent_story.key} has no attachments.")

    else:
        print(f"\nNo user stories were fetched from project {JIRA_PROJECT_KEY} to demonstrate attachment download.")
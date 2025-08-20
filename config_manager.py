# config_manager.py
import configparser
import logging
import os
from typing import Dict, Any, Optional

def read_all_config(config_file_path: str = 'config.ini') -> Optional[Dict[str, Any]]:
    """
    Reads all configurations from the specified INI file.
    Returns a dictionary with configuration values or None if critical errors occur.
    """
    # --- START OF DEBUGGING PRINTS ---
    print(f"--- [DEBUG] Attempting to read config from: {config_file_path}")
    
    # Check if the file exists at the given path
    if not os.path.exists(config_file_path):
        print(f"--- [DEBUG] ERROR: File does not exist at path: {config_file_path}")
        logging.error(f"Configuration file '{config_file_path}' not found.")
        return None
    
    print(f"--- [DEBUG] SUCCESS: Config file found at path: {config_file_path}")
    # --- END OF DEBUGGING PRINTS ---

    config = configparser.ConfigParser()
    config_values: Dict[str, Any] = {}

    try:
        config.read(config_file_path)

        # --- MORE DEBUGGING ---
        print(f"--- [DEBUG] Sections found in config file: {config.sections()}")
        # --- END OF DEBUGGING ---

        # [DEFAULT] section
        if config.has_section('DEFAULT'):
            print("--- [DEBUG] Found [DEFAULT] section. Reading keys...")
            config_values['project_id'] = config.get('DEFAULT', 'project_id', fallback=None)
            config_values['input_csv_file_path'] = config.get('DEFAULT', 'input_csv_file_path', fallback=None)
            config_values['output_csv_file_path'] = config.get('DEFAULT', 'output_csv_file_path', fallback=None)
            config_values['output_sql_file'] = config.get('DEFAULT', 'output_sql_file', fallback=None)
            config_values['learning_repo_csv_path'] = config.get('DEFAULT', 'learning_repo_csv_path', fallback=None)
            config_values['loop_developer_max_retries'] = config.getint('DEFAULT', 'loop_developer_max_retries', fallback=3)
        else:
            print("--- [DEBUG] WARNING: [DEFAULT] section NOT FOUND in config file.")

        # [DATABASE] section
        if config.has_section('DATABASE'):
            print("--- [DEBUG] Found [DATABASE] section. Reading keys...")
            config_values['project_id'] = config.get('DATABASE', 'project_id', fallback=None)
            config_values['input_csv_file_path'] = config.get('DATABASE', 'input_csv_file_path', fallback=None)
            config_values['output_csv_file_path'] = config.get('DATABASE', 'output_csv_file_path', fallback=None)
            config_values['output_sql_file'] = config.get('DATABASE', 'output_sql_file', fallback=None)
            config_values['learning_repo_csv_path'] = config.get('DATABASE', 'learning_repo_csv_path', fallback=None)
            config_values['loop_developer_max_retries'] = config.getint('DATABASE', 'loop_developer_max_retries', fallback=3)
            config_values['db_inst_uri'] = config.get('DATABASE', 'db_inst_uri', fallback=None)
            config_values['db_user'] = config.get('DATABASE', 'db_user', fallback=None)
            config_values['db_password'] = config.get('DATABASE', 'db_password', fallback=None)
            config_values['db_name'] = config.get('DATABASE', 'db_name', fallback=None)

        # [VERTEX_AI] section
        if config.has_section('VERTEX_AI'):
            config_values['vertex_ai_location'] = config.get('VERTEX_AI', 'location', fallback=None)
            config_values['vertex_ai_model_name'] = config.get('VERTEX_AI', 'model', fallback=None)
        
        # [JIRA] section
        if config.has_section('JIRA'):
            config_values['jira_server'] = config.get('JIRA', 'server', fallback=None)
            config_values['jira_email'] = config.get('JIRA', 'email', fallback=None)
            config_values['jira_api_token'] = config.get('JIRA', 'api_token', fallback=None)
            config_values['jira_project_key'] = config.get('JIRA', 'project_key', fallback=None)
            
        # [GEMINI_API] section
        if config.has_section('GEMINI_API'):
            config_values['gemini_api_key'] = config.get('GEMINI_API', 'api_key', fallback=None)

        # --- Final Validation ---
        if not config_values.get('project_id'):
            print("--- [DEBUG] Final validation failed. 'project_id' key not found in the config_values dictionary.")
            logging.error("CRITICAL: 'project_id' is missing from the [DEFAULT] section in config.ini")
            print("CRITICAL: 'project_id' is missing from the [DEFAULT] section in config.ini")
            return None

        print("--- [DEBUG] Final validation passed. 'project_id' was loaded.")
        return config_values

    except Exception as e:
        logging.error(f"An unexpected error occurred while reading config file '{config_file_path}': {e}", exc_info=True)
        print(f"Error: Unexpected error reading config file '{config_file_path}'. Error: {e}. Exiting.")
        return None
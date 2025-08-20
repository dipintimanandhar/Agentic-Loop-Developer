# app_logic.py
import logging
import pandas as pd
import csv
import pg8000
import sqlalchemy
from sqlalchemy.exc import SQLAlchemyError
from google.cloud.alloydb.connector import Connector
from vertexai.generative_models import GenerativeModel, Part # Part kept for potential future use
import datetime
import re
import json
import os
import ast
import math
from typing import Tuple, Optional, List, Dict, Any

import jira_publish # Assumed to be an existing module

# === Python Loop Developer Mode Functions ===

def extract_python_code_from_response(text_response: str) -> str:
    """Extracts Python code from a markdown-formatted string."""
    match = re.search(r"```python\n(.*?)\n```", text_response, re.DOTALL)
    if match:
        return match.group(1).strip()
    match = re.search(r"```\n(.*?)\n```", text_response, re.DOTALL) # Fallback
    if match:
        return match.group(1).strip()
    logging.warning("Loop Developer - Could not find Python code in markdown. Using entire response.")
    return text_response.strip() if text_response else ""

def generate_python_code_for_story(
    model: GenerativeModel,
    story_description: str,
    io_examples: List[Dict[str, str]],
    language: str = "python",
    previous_code: Optional[str] = None,
    error_feedback: Optional[str] = None
) -> Optional[str]:
    """Generates Python code for a given user story using the LLM."""
    prompt_parts = [f"User Story: {story_description}\n"]
    prompt_parts.append(f"Please generate a complete {language} code solution. The primary function MUST be named 'solve_story'.")
    prompt_parts.append("Ensure all necessary imports are included within the function if they are not standard Python libraries, or list them if they are standard (e.g., math, json, re, ast).")

    if io_examples:
        prompt_parts.append("\nInput/output examples (input/output are strings, parse/evaluate them appropriately using 'ast.literal_eval' for safety if they represent Python literals):")
        for ex in io_examples:
            prompt_parts.append(f"- Input: {ex['input']}\n  Output: {ex['output']}")

    if previous_code:
        prompt_parts.append(f"\nPrevious code attempt with issues:\n```python\n{previous_code}\n```")
    if error_feedback:
        prompt_parts.append(f"\nError feedback from last execution: {error_feedback}")
        prompt_parts.append("\nAnalyze the error and code, then provide a corrected 'solve_story' function.")
    else:
        prompt_parts.append("\nProvide the 'solve_story' function based on description and examples.")
    
    prompt_parts.append("\nIMPORTANT: Only output the Python code block for 'solve_story' and helpers/imports. No explanatory text outside code block unless it's comments within code.")
    full_prompt = "\n".join(prompt_parts)
    
    logging.debug(f"Loop Developer - Prompt to Gemini (Story Gen):\n{full_prompt[:500]}...")
    print(f"\n--- Sending Prompt to Gemini (Story Gen) ---\n{full_prompt[:500]}...\n--------------------------------------\n")

    try:
        response = model.generate_content(full_prompt)
        generated_text = ""
        if response.candidates and response.candidates[0].content and response.candidates[0].content.parts:
            generated_text = "".join(part.text for part in response.candidates[0].content.parts if hasattr(part, 'text'))
        elif hasattr(response, 'text'):
             generated_text = response.text
        
        logging.debug(f"Loop Developer - Gemini Raw Response: {generated_text[:300]}...")
        return extract_python_code_from_response(generated_text)
    except Exception as e:
        logging.error(f"Loop Developer - Error during Gemini API call for story generation: {e}", exc_info=True)
        return None

def execute_and_test_python_code(code_string: str, io_examples: List[Dict[str, str]]) -> Tuple[bool, List[Dict[str, Any]], str]:
    """Executes the generated Python code and tests it against I/O examples."""
    results = []
    all_passed = True
    full_error_output = ""

    if not code_string:
        logging.warning("Loop Developer - No code string provided to execute.")
        return False, [{"error": "No code string provided."}], "No code to execute."

    execution_scope = {'ast': ast, 'math': math, 'json': json, 're': re}
    try:
        exec(code_string, execution_scope)
        if 'solve_story' not in execution_scope or not callable(execution_scope['solve_story']):
            raise NameError("Function 'solve_story' not defined or not callable.")
        solution_func = execution_scope['solve_story']

        for ex in io_examples:
            input_str, expected_output_str = ex['input'], ex['output']
            test_passed, actual_output_val, error_in_test = False, "N/A", None
            try:
                actual_input = ast.literal_eval(input_str)
                expected_output = ast.literal_eval(expected_output_str)
                actual_output_val = solution_func(actual_input)
                if actual_output_val == expected_output:
                    test_passed = True
                else:
                    all_passed = False
            except Exception as e:
                all_passed = False
                error_in_test = f"Test Exec Error (Input: {input_str}): {type(e).__name__}: {e}"
                full_error_output += error_in_test + "\n"
                logging.warning(f"Loop Developer - {error_in_test}")
            results.append({
                "input": input_str, "expected": expected_output_str, "actual": str(actual_output_val),
                "passed": test_passed, "error": error_in_test
            })
    except Exception as e:
        all_passed = False
        compilation_error = f"Code Execution/Setup Error: {type(e).__name__}: {e}"
        logging.error(f"Loop Developer - {compilation_error}", exc_info=True)
        full_error_output = compilation_error # Overwrites individual test errors if global setup fails
        results.append({"error": compilation_error, "passed": False, "input": "N/A", "expected": "N/A", "actual": "N/A"})
    
    final_error_msg = full_error_output.strip() if full_error_output else ("Tests failed." if not all_passed else "")
    return all_passed, results, final_error_msg

def process_python_story(story_data: Dict[str, Any], model: GenerativeModel, max_retries: int):
    """Processes a single Python user story: generate, test, and refine code."""
    story_id = story_data.get("id", "Unknown Story")
    summary = story_data.get("summary", "No Summary")
    description = story_data.get("description", "No Description")
    io_examples = story_data.get("io_examples", [])
    language = story_data.get("language", "python")

    logging.info(f"Loop Developer - Processing Story: {story_id} - {summary}")
    print(f"\n\n===================================================")
    print(f"Processing Python Story: {story_id} - {summary}")
    print(f"===================================================\nDescription: {description}")

    if language.lower() != "python":
        logging.warning(f"Loop Developer - Skipping story {story_id} (language: {language}). Only Python supported.")
        print(f"Skipping story {story_id} (language: {language}) - Not Python.")
        return

    current_code, last_error_feedback = None, None
    for attempt in range(max_retries):
        logging.info(f"Loop Developer - Story {story_id}, Attempt {attempt + 1}/{max_retries}")
        print(f"\n--- Story {story_id}: Attempt {attempt + 1}/{max_retries} ---")
        
        current_code = generate_python_code_for_story(
            model, description, io_examples, language,
            previous_code=current_code, error_feedback=last_error_feedback
        )

        if not current_code:
            logging.error(f"Loop Developer - Story {story_id}: No code from Gemini on attempt {attempt + 1}.")
            last_error_feedback = "Gemini did not return code or an API error occurred."
            if attempt == max_retries - 1:
                logging.error(f"Loop Developer ({story_id}): FINAL FAILURE - No code after {max_retries} attempts.")
                print(f"Loop Developer ({story_id}): FINAL FAILURE - No code generated after {max_retries} attempts.")
            continue
        
        print(f"\n--- Generated Python Code (Attempt {attempt + 1}) ---\n{current_code}\n-------------------------------------\n")
        all_tests_passed, test_results, execution_errors = execute_and_test_python_code(current_code, io_examples)
        
        print("\n--- Test Results ---")
        for res in test_results:
            status_color = "\033[92m" if res['passed'] else "\033[91m"
            print(f"  Input: {res['input']}, Expected: {res['expected']}, Actual: {res['actual']}, {status_color}Passed: {res['passed']}\033[0m")
            if res.get('error'): print(f"    Error: {res['error']}")
        
        if execution_errors and not any(r.get("error") for r in test_results): # Display global error if no specific test error was already shown
             print(f"\n  Overall Execution/Compilation Error: {execution_errors}")

        if all_tests_passed:
            logging.info(f"Loop Developer ({story_id}): SUCCESS on attempt {attempt + 1}.")
            print(f"\033[92m\nLoop Developer ({story_id}): SUCCESS! All tests passed on attempt {attempt + 1}.\033[0m")
            print(f"Final working Python code for {story_id}:\n{current_code}")
            # Here you might want to save the code or update Jira
            return
        else:
            logging.warning(f"Loop Developer ({story_id}): Tests failed on attempt {attempt + 1}.")
            print(f"\033[91m\nLoop Developer ({story_id}): Some tests failed on attempt {attempt + 1}.\033[0m")
            
            # Construct detailed feedback
            error_details = []
            if execution_errors:
                error_details.append(f"Overall Execution/Compilation Error: {execution_errors}")
            for res in test_results:
                if not res['passed']:
                    detail = f"- Input: {res['input']}, Expected: {res['expected']}, Got: {res['actual']}"
                    if res.get('error'): detail += f" (Test Error: {res['error']})"
                    error_details.append(detail)
            last_error_feedback = "Some tests failed.\n" + "\n".join(error_details)

            if attempt < max_retries - 1:
                print("Will attempt to refine...")
            else:
                logging.error(f"Loop Developer ({story_id}): FINAL FAILURE after {max_retries} attempts.")
                print(f"Loop Developer ({story_id}): FINAL FAILURE after {max_retries} attempts.")
                print(f"Last attempted Python code for {story_id}:\n{current_code}")
                print(f"Last error feedback for {story_id}:\n{last_error_feedback}")

def run_loop_developer_mode(config_params: Dict[str, Any], model: GenerativeModel):
    """Runs the Loop Developer mode to generate Python code from Jira user stories."""
    logging.info("Starting Loop Developer Mode...")
    print("\n--- Loop Developer Mode: Generating Python Code from User Stories ---")
    
    max_retries = config_params.get('loop_developer_max_retries', 3)

    try:
        stories = jira_publish.fetch_existing_user_stories() # Assumes this function is well-defined
    except Exception as e:
        logging.error(f"Loop Developer - Failed to fetch Jira stories: {e}", exc_info=True)
        print(f"Error: Could not fetch Jira stories: {e}")
        return

    if not stories:
        logging.warning("Loop Developer - No Jira stories found or fetched.")
        print("No Jira stories found to process.")
        return
    
    print(f"Found {len(stories)} Jira stories. Processing Python-related stories...")
    for issue in stories:
        # Example: Filter for stories that might be Python related based on summary/labels (customize as needed)
        # For this example, we'll assume all fetched stories might be Python if they have a description.
        # A more robust filter might check for specific labels or keywords.
        
        summary = issue.fields.summary if issue.fields.summary else "No Summary"
        description = issue.fields.description if issue.fields.description else ""
        
        # Basic filter: if "python" is in summary or description, or if no specific language is mentioned assume python.
        # This is a placeholder; you'll need more robust story selection criteria.
        is_python_story = "python" in summary.lower() or "python" in description.lower() if description else "python" in summary.lower()

        if not is_python_story: # Simple heuristic, improve as needed
             # Try to parse io_examples from description if they exist in a structured format
            io_examples_parsed = []
            if description:
                # This is a placeholder for robust I/O example parsing from Jira description
                # e.g., looking for "Examples:\n- Input: ... Output: ..."
                # For now, we assume io_examples might be in a custom field or need manual definition
                # For the purpose of this refactor, we keep it simple and rely on story_data potentially having it.
                pass


        # Create the story_data dictionary
        story_data_dict = {
            "id": issue.key,
            "summary": summary,
            "description": description if description else "No description provided.",
            "io_examples": io_examples_parsed, # Use parsed examples or expect them to be added/found elsewhere
            "language": "python", # Defaulting to python, could be refined by labels/tags
            "status": issue.fields.status.name if issue.fields.status else "Unknown Status"
        }
        
        # If no I/O examples are found in description, prompt user or use defaults
        if not story_data_dict["io_examples"]:
            logging.warning(f"Loop Developer - Story {issue.key}: No I/O examples found in description. Code generation quality may be affected.")
            # In a real scenario, you might skip, prompt, or have a way to define these.
            # For now, we proceed, but the LLM might struggle without examples.
            # Example: ask user for I/O examples or use placeholders if necessary for the demo
            # For simplicity, we'll allow proceeding without examples, but log it.
            print(f"Warning: Story {issue.key} has no I/O examples from description. LLM will try without them.")


        process_python_story(story_data_dict, model, max_retries)

    logging.info("Loop Developer Mode finished.")
    print("\n--- Loop Developer Mode Finished ---")


# === SQL Code Conversion Mode Functions ===

def create_sqlalchemy_engine(
    inst_uri: str, user: str, password: str, db: str,
    refresh_strategy: str = "background",
) -> Tuple[Optional[sqlalchemy.engine.Engine], Optional[Connector]]:
    """Creates an SQLAlchemy engine for AlloyDB."""
    try:
        logging.info(f"SQL Mode - Creating AlloyDB connector for {inst_uri}...")
        connector = Connector(refresh_strategy=refresh_strategy)
        
        def getconn() -> pg8000.dbapi.Connection:
            conn_obj: pg8000.dbapi.Connection = connector.connect(
                instance_uri=inst_uri, driver="pg8000", user=user, password=password, db=db,
            )
            return conn_obj
        
        engine = sqlalchemy.create_engine("postgresql+pg8000://", creator=getconn, echo=False)
        logging.info(f"SQLAlchemy engine created for {inst_uri}/{db}")
        return engine, connector
    except Exception as e:
        logging.error(f"SQL Mode - Failed to create SQLAlchemy engine for {inst_uri}: {e}", exc_info=True)
        print(f"SQL Mode - Error: Could not connect to database. Check details/proxy. Error: {e}")
        return None, None

def _clean_llm_sql_response(raw_text: str) -> str:
    """Cleans the raw SQL response from LLM."""
    cleaned = raw_text.strip()
    if cleaned.startswith("```sql"): cleaned = cleaned[len("```sql"):].strip()
    elif cleaned.startswith("```"): cleaned = cleaned[len("```"):].strip()
    if cleaned.endswith("```"): cleaned = cleaned[:-len("```")].strip()
    if cleaned.endswith(";"): cleaned = cleaned[:-1].strip() # Remove trailing semicolon
    return cleaned

def _execute_single_pg_query_with_retry(
    pg_query: str, oracle_query: str, conn: sqlalchemy.engine.Connection,
    model: GenerativeModel, use_feedback_loop: bool, original_row_idx: int
) -> Tuple[str, Optional[datetime.timedelta], str]:
    """Executes a single PostgreSQL query, with optional retry loop using LLM feedback."""
    current_pg_query = pg_query.strip()
    max_attempts = 3 if use_feedback_loop else 1
    
    if not current_pg_query:
        logging.warning(f"SQL Mode - Row {original_row_idx}: Skipping execution, PostgreSQL query is empty.")
        return "Skipped (Empty)", None, "Converted query was empty."

    for attempt in range(max_attempts):
        logging.info(f"SQL Mode - Row {original_row_idx}, Oracle: {oracle_query[:100]}..., PG (Attempt {attempt+1}/{max_attempts}): {current_pg_query[:100]}...")
        print(f"\n----------------------------------------")
        print(f"Original Oracle Query (Row {original_row_idx}):\n{oracle_query}")
        print(f"Attempting PostgreSQL Query (Attempt {attempt+1}/{max_attempts}):\n{current_pg_query}")
        
        trans: Optional[sqlalchemy.engine.Transaction] = None
        try:
            start_time = datetime.datetime.now()
            trans = conn.begin()
            conn.execute(sqlalchemy.text(current_pg_query))
            trans.commit()
            end_time = datetime.datetime.now()
            exec_time = end_time - start_time
            logging.info(f"SQL Mode - Row {original_row_idx} (Attempt {attempt+1}) SUCCESS. Time: {exec_time}")
            print("\033[92m" + f"Query for row {original_row_idx} (Attempt {attempt+1}) executed successfully!" + "\033[0m")
            return "Success", exec_time, ""
        except SQLAlchemyError as e: # Catch specific SQLAlchemy errors
            if trans and trans.is_active: trans.rollback()
            error_message = str(e).replace('\n', ' ').strip()
            logging.warning(f"SQL Mode - Row {original_row_idx} (Attempt {attempt+1}) FAILED: {error_message}")
            print("\033[91m" + f"Query for row {original_row_idx} (Attempt {attempt+1}) failed. Error: {error_message}" + "\033[0m")

            if use_feedback_loop and attempt < max_attempts - 1:
                print(f"Attempting LLM feedback to correct query (Next attempt {attempt+2})...")
                retry_prompt = (
                    f"The following PostgreSQL query failed: {error_message}\n"
                    f"Original Oracle query:\n{oracle_query}\n\n"
                    f"Failed PostgreSQL query:\n{current_pg_query}\n\n"
                    "Provide ONLY the corrected PostgreSQL query. No ```sql, comments, explanations, line numbers, or trailing semicolons."
                )
                try:
                    logging.debug(f"SQL Mode - Row {original_row_idx}: Sending retry prompt to Gemini.")
                    retry_response = model.generate_content(retry_prompt)
                    corrected_query_text = ""
                    if retry_response.candidates and retry_response.candidates[0].content and retry_response.candidates[0].content.parts:
                         corrected_query_text = "".join(part.text for part in retry_response.candidates[0].content.parts if hasattr(part, 'text'))
                    elif hasattr(retry_response, 'text'): # Fallback
                        corrected_query_text = retry_response.text
                    
                    cleaned_correction = _clean_llm_sql_response(corrected_query_text)
                    if cleaned_correction and cleaned_correction != current_pg_query:
                        current_pg_query = cleaned_correction
                        logging.info(f"SQL Mode - Row {original_row_idx}: Feedback loop provided new query: '{current_pg_query[:100]}...'")
                        continue # Retry with the new query
                    else:
                        logging.warning(f"SQL Mode - Row {original_row_idx}: Feedback loop provided empty or unchanged query. Stopping retries for this query.")
                        return "Fail", None, f"Execution failed: {error_message}. Feedback loop did not yield a new query."
                except Exception as llm_retry_e:
                    logging.error(f"SQL Mode - Row {original_row_idx}: Error during LLM retry call: {llm_retry_e}", exc_info=True)
                    return "Fail", None, f"Execution failed: {error_message}. Error in LLM feedback: {str(llm_retry_e)}"
            return "Fail", None, f"Execution failed: {error_message}" # Failed and no more retries or feedback loop not enabled/successful
        except Exception as e_other: # Catch other unexpected errors during execution
            if trans and trans.is_active: trans.rollback()
            error_message = str(e_other).replace('\n', ' ').strip()
            logging.error(f"SQL Mode - Row {original_row_idx} (Attempt {attempt+1}) UNEXPECTED FAILED: {error_message}", exc_info=True)
            print("\033[91m" + f"Query for row {original_row_idx} (Attempt {attempt+1}) failed with unexpected error: {error_message}" + "\033[0m")
            return "Fail", None, f"Unexpected execution error: {error_message}"
            
    return "Fail", None, "Max attempts reached or feedback loop exhausted." # Should be covered by returns inside loop


def _process_sql_block_for_execution(
    pg_query_to_execute: str, current_original_row_index: int, input_df: pd.DataFrame,
    conn: sqlalchemy.engine.Connection, model: GenerativeModel, use_feedback_loop: bool,
    writer: csv.writer
) -> Tuple[int, int, int]:
    """Helper to process a single SQL block, execute it, and write results."""
    successful, failed, skipped = 0, 0, 0
    oracle_query_text = "Oracle query N/A (index OOB)"
    if 0 <= current_original_row_index < len(input_df):
        oracle_query_text = str(input_df.iloc[current_original_row_index].get('oracle', 'Oracle column missing'))
    else:
        logging.warning(f"SQL Mode - Original row index {current_original_row_index} is OOB for input DataFrame during execution.")

    status, exec_time, err_msg = _execute_single_pg_query_with_retry(
        pg_query_to_execute, oracle_query_text, conn, model, use_feedback_loop, current_original_row_index
    )
    exec_time_str = f"{exec_time.total_seconds():.4f}" if exec_time else "N/A"
    writer.writerow([current_original_row_index, pg_query_to_execute, status, exec_time_str, err_msg])

    if status == "Success": successful = 1
    elif status.startswith("Skipped"): skipped = 1
    else: failed = 1
    return successful, failed, skipped

def execute_sql_file_and_store_results(
    sql_file_path: str, input_df: pd.DataFrame, output_csv_file_path: str,
    config_params: Dict[str, Any], model: GenerativeModel, use_feedback_loop: bool
):
    """Executes SQL queries from a file and stores results in a CSV."""
    engine, connector = None, None
    db_conf_keys = ['inst_uri', 'db_user', 'db_password', 'db_name']
    if not all(config_params.get(key) for key in db_conf_keys):
        logging.error("SQL Mode - Missing database connection details in config for execution.")
        print("Error: SQL execution requires inst_uri, db_user, db_password, db_name in config.")
        return
        
    successful_executions, failed_executions, skipped_empty_executions = 0, 0, 0

    try:
        engine, connector = create_sqlalchemy_engine(
            config_params['inst_uri'], config_params['db_user'], 
            config_params['db_password'], config_params['db_name']
        )
        if not engine: # Connector presence checked by engine creation success
            logging.critical("SQL Mode - DB engine creation failed. Cannot execute queries.")
            return
        
        output_csv_file_path = 'jira_attachments/KAN-10/output_execution_results.csv'

        with engine.connect() as conn, open(output_csv_file_path, "w", newline="", encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["original_oracle_query_row_index", "converted_postgresql_query", "execution_status", "execution_time_seconds", "error_message"])
            
            current_sql_block_lines: List[str] = []
            current_original_row_index: int = -1

            try:
                with open(sql_file_path, "r", encoding='utf-8') as file:
                    for line_num, line_content in enumerate(file, 1):
                        stripped_line = line_content.strip()
                        if stripped_line.startswith("-- Converted query for row"):
                            if current_sql_block_lines and current_original_row_index != -1:
                                pg_query_to_execute = "\n".join(current_sql_block_lines).strip()
                                s, f, sk = _process_sql_block_for_execution(
                                    pg_query_to_execute, current_original_row_index, input_df,
                                    conn, model, use_feedback_loop, writer
                                )
                                successful_executions += s; failed_executions += f; skipped_empty_executions += sk
                            
                            current_sql_block_lines = [] # Reset for next block
                            try:
                                match = re.search(r"row (\d+)", stripped_line)
                                current_original_row_index = int(match.group(1)) if match else -1
                                if current_original_row_index == -1:
                                     logging.warning(f"SQL Mode - File '{sql_file_path}', line {line_num}: Could not parse row index: '{stripped_line}'")
                            except ValueError:
                                logging.error(f"SQL Mode - File '{sql_file_path}', line {line_num}: Error parsing index: '{stripped_line}'", exc_info=True)
                                current_original_row_index = -1
                            continue # Move to next line
                        
                        if current_original_row_index != -1 and not stripped_line.startswith("--"):
                            current_sql_block_lines.append(line_content.rstrip('\n'))

                    # Process the last block in the file
                    if current_sql_block_lines and current_original_row_index != -1:
                        pg_query_to_execute = "\n".join(current_sql_block_lines).strip()
                        s, f, sk = _process_sql_block_for_execution(
                            pg_query_to_execute, current_original_row_index, input_df,
                            conn, model, use_feedback_loop, writer
                        )
                        successful_executions += s; failed_executions += f; skipped_empty_executions += sk

            except FileNotFoundError:
                logging.error(f"SQL Mode - SQL file not found: {sql_file_path}")
                print(f"Error: SQL file '{sql_file_path}' not found.")
            except Exception as e_file:
                logging.error(f"SQL Mode - Error reading/processing SQL file {sql_file_path}: {e_file}", exc_info=True)
                print(f"Error: Could not process SQL file '{sql_file_path}'. Error: {e_file}")

    except SQLAlchemyError as e_db:
        logging.error(f"SQL Mode - DB connection or query error: {e_db}", exc_info=True)
        print(f"Critical Error: Database connection/execution failed. Error: {e_db}")
    except IOError as e_io:
        logging.error(f"SQL Mode - File I/O error with '{output_csv_file_path}': {e_io}", exc_info=True)
        print(f"Critical Error: Could not write to output CSV '{output_csv_file_path}'. Error: {e_io}")
    except Exception as e_main:
        logging.error(f"SQL Mode - Unexpected error in execute_sql_file_and_store_results: {e_main}", exc_info=True)
        print(f"Critical Error: An unexpected error occurred during SQL execution. Error: {e_main}")
    finally:
        if connector:
            connector.close()
            logging.info("SQL Mode - AlloyDB connector closed.")

    print("\n--- SQL Execution Outcomes ---")
    total_processed = successful_executions + failed_executions + skipped_empty_executions
    print(f"Total query blocks processed: {total_processed}")
    print(f"Successful executions: {successful_executions}")
    print(f"Failed executions: {failed_executions}")
    if skipped_empty_executions > 0: print(f"Skipped (empty query): {skipped_empty_executions}")
    print(f"Output CSV report: {output_csv_file_path}")


def run_sql_conversion_mode(config_params: Dict[str, Any], model: GenerativeModel):
    """Runs the SQL Code Conversion mode (Oracle to PostgreSQL)."""
    logging.info("Starting SQL Code Conversion Mode...")
    print("\n--- SQL Code Conversion Mode (Oracle to PostgreSQL) ---")

    required_conf = ['inst_uri', 'db_user', 'db_password', 'db_name'] # For execution part
    # input_csv, output_csv, output_sql are now sourced from Jira attachments per story
    
    learning_repo_csv = config_params.get('learning_repo_csv_path')

    use_learning_repo, use_feedback_loop = False, False
    try:
        lr_input = input("Use learning repository for SQL conversion guidance? (yes/no, default no): ").lower().strip()
        use_learning_repo = lr_input == "yes"
        fl_input = input("Use feedback loop for failed SQL queries during execution? (yes/no, default no): ").lower().strip()
        use_feedback_loop = fl_input == "yes"
    except KeyboardInterrupt:
        logging.info("SQL Mode - User interrupted setup. Exiting mode.")
        print("\nOperation cancelled by user.")
        return
    logging.info(f"SQL Mode Settings: Learning Repo: {use_learning_repo}, Feedback Loop: {use_feedback_loop}")

    rule_guidance_text = ""
    if use_learning_repo and learning_repo_csv:
        try:
            lr_df = pd.read_csv(learning_repo_csv)
            if "Rule_guidance" in lr_df.columns:
                valid_guidance = lr_df["Rule_guidance"].dropna().astype(str).str.strip()
                rule_guidance_text = "\n".join(g for g in valid_guidance.tolist() if g) # Filter empty strings
                if rule_guidance_text: logging.info(f"SQL Mode - Loaded learning guidance from {learning_repo_csv}")
                else: logging.info(f"SQL Mode - Learning repo {learning_repo_csv} 'Rule_guidance' is empty.")
            else: logging.warning(f"SQL Mode - 'Rule_guidance' col not in {learning_repo_csv}.")
        except FileNotFoundError: logging.error(f"SQL Mode - Learning repo not found: {learning_repo_csv}.")
        except pd.errors.EmptyDataError: logging.warning(f"SQL Mode - Learning repo {learning_repo_csv} is empty.")
        except Exception as e: logging.error(f"SQL Mode - Error reading learning repo {learning_repo_csv}: {e}", exc_info=True)
    
    try:
        stories = jira_publish.fetch_existing_user_stories()
    except Exception as e:
        logging.error(f"SQL Mode - Failed to fetch Jira stories: {e}", exc_info=True)
        print(f"Error: Could not fetch Jira stories for SQL conversion: {e}")
        return

    if not stories:
        logging.warning("SQL Mode - No Jira stories found for SQL conversion.")
        print("No Jira stories found to process for SQL conversion.")
        return

    processed_any_story = False
    for issue in stories:
        # Filter for SQL conversion stories (e.g., based on summary, labels)
        summary_lower = issue.fields.summary.lower() if issue.fields.summary else ""
        if not ('sql' in summary_lower and 'convert' in summary_lower): # Basic filter
            continue
        
        processed_any_story = True
        print(f"\nProcessing SQL Conversion Story: {issue.key}: {issue.fields.summary}")
        logging.info(f"SQL Mode - Processing story {issue.key} for SQL conversion.")

        # Define attachment paths based on Jira issue key
        # Ensure 'jira_attachments' and subdirectories are writable/creatable
        base_attachment_path = os.path.join('jira_attachments', issue.key)
        os.makedirs(base_attachment_path, exist_ok=True)

        input_csv_path = os.path.join(base_attachment_path, 'input_sql.csv')
        output_sql_path = os.path.join(base_attachment_path, 'output_sql.sql')
        output_exec_csv_path = os.path.join(base_attachment_path, 'output_execution_results.csv')

        try:
            # Assuming download_attachments_from_story saves 'input_sql.csv' to input_csv_path
            jira_publish.download_attachments_from_story(issue.key)
            if not os.path.exists(input_csv_path):
                logging.warning(f"SQL Mode - Story {issue.key}: 'input_sql.csv' not found after download attempt. Skipping story.")
                print(f"Warning: Story {issue.key} - 'input_sql.csv' not found. Cannot proceed with this story.")
                continue
        except Exception as e_jira_dl:
            logging.error(f"SQL Mode - Story {issue.key}: Failed to download attachments: {e_jira_dl}", exc_info=True)
            print(f"Error: Story {issue.key} - Failed to download attachments: {e_jira_dl}. Skipping story.")
            continue
            
        input_df = pd.DataFrame()
        try:
            input_df = pd.read_csv(input_csv_path)
            if 'oracle' not in input_df.columns:
                logging.error(f"SQL Mode - Story {issue.key}: 'oracle' column not in {input_csv_path}.")
                print(f"Error: Story {issue.key} - 'oracle' column missing in {input_csv_path}. Skipping.")
                continue
            if input_df.empty:
                logging.info(f"SQL Mode - Story {issue.key}: Input CSV {input_csv_path} is empty.")
                print(f"Info: Story {issue.key} - Input CSV {input_csv_path} is empty. No queries to convert.")
                # Potentially upload an empty output_sql.sql and output_exec_csv.csv or just skip
                # For now, we continue to the next story.
                with open(output_sql_path, 'w', encoding='utf-8') as f_sql_empty:
                    f_sql_empty.write(f"-- No Oracle queries found in {input_csv_path} for story {issue.key}.\n")
                jira_publish.upload_attachment_to_story(issue.key, output_sql_path)
                # Create an empty exec results CSV as well for consistency
                with open(output_exec_csv_path, 'w', newline='', encoding='utf-8') as f_csv_empty:
                    writer = csv.writer(f_csv_empty)
                    writer.writerow(["original_oracle_query_row_index", "converted_postgresql_query", "execution_status", "execution_time_seconds", "error_message"])
                    writer.writerow([-1, "-- No queries processed --", "Skipped", "N/A", "Input CSV was empty or lacked 'oracle' column."])
                jira_publish.upload_attachment_to_story(issue.key, output_exec_csv_path)
                continue

            logging.info(f"SQL Mode - Story {issue.key}: Loaded {len(input_df)} Oracle queries from {input_csv_path}.")
        except Exception as e_read_csv:
            logging.error(f"SQL Mode - Story {issue.key}: Error reading {input_csv_path}: {e_read_csv}", exc_info=True)
            print(f"Error: Story {issue.key} - Could not read {input_csv_path}: {e_read_csv}. Skipping.")
            continue

        conversion_errors = 0
        try:
            with open(output_sql_path, 'w', encoding='utf-8') as f_sql_out:
                f_sql_out.write(f"-- Converted PostgreSQL queries for Jira Story: {issue.key}\n")
                f_sql_out.write(f"-- Source Oracle CSV: {os.path.basename(input_csv_path)}\n")
                f_sql_out.write(f"-- Conversion Date: {datetime.datetime.now().isoformat()}\n\n")

                for index, row in input_df.iterrows():
                    oracle_query = str(row.get('oracle', '')).strip() # Use .get for safety
                    if not oracle_query:
                        logging.warning(f"SQL Mode - Story {issue.key}, Row {index}: Oracle query empty. Skipping conversion.")
                        f_sql_out.write(f"-- Converted query for original CSV row {index}\n-- Original Oracle query was empty.\n\n")
                        continue
                    
                    prompt_parts = [
                        "Convert the following Oracle SQL to precise PostgreSQL equivalent.\n",
                        "Oracle SQL:\n```sql\n", oracle_query, "\n```\n"
                    ]
                    if rule_guidance_text:
                        prompt_parts.extend(["Apply this guidance if relevant:\n", rule_guidance_text, "\n"])
                    prompt_parts.extend([
                        "Respond ONLY with the converted PostgreSQL query. No ```sql markers, no comments unless part of SQL, no explanations, no line numbers, no trailing semicolons."
                    ])
                    prompt = "".join(prompt_parts) # Use join for efficiency

                    try:
                        logging.info(f"SQL Mode - Story {issue.key}, Row {index}: Converting query...")
                        response = model.generate_content(prompt)
                        raw_response_text = "".join(part.text for part in response.candidates[0].content.parts if hasattr(part, 'text')) if response.candidates and response.candidates[0].content.parts else (response.text if hasattr(response, 'text') else "")
                        
                        cleaned_query = _clean_llm_sql_response(raw_response_text)
                        f_sql_out.write(f"-- Converted query for original CSV row {index}\n")
                        if cleaned_query:
                            f_sql_out.write(cleaned_query + "\n\n")
                        else:
                            f_sql_out.write(f"-- Conversion resulted in empty query for original CSV row {index}.\n")
                            f_sql_out.write("-- Original Oracle Query (commented out):\n")
                            for line in oracle_query.splitlines(): f_sql_out.write(f"-- {line}\n")
                            f_sql_out.write("\n\n")
                            conversion_errors +=1
                            logging.warning(f"SQL Mode - Story {issue.key}, Row {index}: Conversion empty. Oracle: {oracle_query[:100]}...")
                    except Exception as e_conv:
                        conversion_errors += 1
                        logging.error(f"SQL Mode - Story {issue.key}, Row {index}: Error converting query: {e_conv}", exc_info=True)
                        f_sql_out.write(f"-- Conversion failed for original CSV row {index}: {str(e_conv).replacechr(10),' '}\n") # Quick sanitize
                        f_sql_out.write("-- Original Oracle Query (commented out):\n")
                        for line in oracle_query.splitlines(): f_sql_out.write(f"-- {line}\n")
                        f_sql_out.write("\n\n")
            
            logging.info(f"SQL Mode - Story {issue.key}: Converted queries saved to '{output_sql_path}'. Conversion errors: {conversion_errors}")
            jira_publish.upload_attachment_to_story(issue.key, output_sql_path)

        except IOError as e_io_sql:
            logging.error(f"SQL Mode - Story {issue.key}: Could not write to '{output_sql_path}': {e_io_sql}", exc_info=True)
            print(f"Error: Story {issue.key} - Could not write to '{output_sql_path}'. Skipping execution for this story.")
            continue # Skip execution if SQL file couldn't be written
        except Exception as e_main_conv:
            logging.error(f"SQL Mode - Story {issue.key}: Unexpected error during SQL conversion: {e_main_conv}", exc_info=True)
            print(f"Error: Story {issue.key} - Unexpected error during SQL conversion. Skipping execution.")
            continue

        # Proceed to execution only if SQL file was generated and there's something to execute
        has_executable_queries = False
        if os.path.exists(output_sql_path):
            try:
                with open(output_sql_path, 'r', encoding='utf-8') as f_check:
                    has_executable_queries = any(not line.strip().startswith("--") and line.strip() for line in f_check)
            except Exception as e_check:
                 logging.error(f"SQL Mode - Story {issue.key}: Error checking '{output_sql_path}' for executable queries: {e_check}", exc_info=True)


        if has_executable_queries:
            if not all(config_params.get(key) for key in required_conf):
                logging.error(f"SQL Mode - Story {issue.key}: Missing DB config for execution. Skipping execution.")
                print(f"Error: Story {issue.key} - DB connection details missing. Cannot execute SQL. Ensure config.ini has DATABASE section correctly set.")
            else:
                logging.info(f"SQL Mode - Story {issue.key}: Executing converted queries from '{output_sql_path}'...")
                execute_sql_file_and_store_results(
                    output_sql_path, input_df, output_exec_csv_path, 
                    config_params, model, use_feedback_loop
                )
                jira_publish.upload_attachment_to_story(issue.key, output_exec_csv_path)
        else:
            logging.info(f"SQL Mode - Story {issue.key}: No executable queries in '{output_sql_path}'. Skipping execution.")
            # Create an empty/informative execution results CSV
            with open(output_exec_csv_path, 'w', newline='', encoding='utf-8') as f_csv_no_exec:
                writer = csv.writer(f_csv_no_exec)
                writer.writerow(["original_oracle_query_row_index", "converted_postgresql_query", "execution_status", "execution_time_seconds", "error_message"])
                writer.writerow([-1, "-- No executable queries generated --", "Skipped", "N/A", f"File '{os.path.basename(output_sql_path)}' had no queries to run."])
            jira_publish.upload_attachment_to_story(issue.key, output_exec_csv_path)

    if not processed_any_story:
        print("No Jira stories matched the criteria for SQL conversion.")

    logging.info("SQL Code Conversion Mode finished.")
    print("\n--- SQL Code Conversion Mode Finished ---")
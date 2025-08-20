# developer_logic.py

import logging
import pandas as pd
import os
import re
import ast
from typing import List, Dict

from vertexai.generative_models import GenerativeModel
from adk_developer_agent.tools.jira_tools import download_attachments, upload_attachment

# ==============================================================================
# === PYTHON STORY LOGIC (This remains unchanged) ==============================
# ==============================================================================

def _extract_python_code(response_text: str) -> str:
    match = re.search(r"```python\n(.*?)\n```", response_text, re.DOTALL)
    return match.group(1).strip() if match else response_text.strip()

def _execute_and_test_code(code_string: str, io_examples: List[Dict]) -> tuple[bool, str]:
    try:
        exec_scope = {'ast': ast}
        exec(code_string, exec_scope)
        solve_story = exec_scope.get('solve_story')
        if not callable(solve_story):
            return False, "Execution Error: 'solve_story' function not found or is not callable."

        failed_tests = []
        for ex in io_examples:
            try:
                actual_input = ast.literal_eval(ex['input'])
                expected_output = ast.literal_eval(ex['output'])
                actual_output = solve_story(actual_input)
                if actual_output != expected_output:
                    failed_tests.append(f"- FAILED: For input `{ex['input']}`, expected `{ex['output']}`, but got `{actual_output}`.")
            except Exception as e:
                failed_tests.append(f"- FAILED: Test execution for input `{ex['input']}` raised an error: {e}")
        
        if not failed_tests:
            return True, "All tests passed!"
        else:
            return False, "Some tests failed:\n" + "\n".join(failed_tests)
    except Exception as e:
        return False, f"Code failed to execute or compile. Error: {e}"

def process_single_python_story(issue_key: str, config: dict, model: GenerativeModel) -> str:
    print(f"Starting Python 'Loop Developer' process for {issue_key}...")
    description = f"This is the user story description for {issue_key}. It needs a Python solution."
    io_examples = [{'input': "'hello'", 'output': "'HELLO'"}, {'input': "'world'", 'output': "'WORLD'"}]
    max_retries = config.get('loop_developer_max_retries', 3)
    current_code, feedback = None, None

    for attempt in range(max_retries):
        print(f"Attempt {attempt + 1}/{max_retries} for {issue_key}...")
        prompt = f"User Story: {description}\nGenerate a complete Python code solution. The primary function must be named 'solve_story'."
        if io_examples: prompt += "\nInput/Output Examples:\n" + "\n".join([f"- Input: {ex['input']}, Output: {ex['output']}" for ex in io_examples])
        if current_code: prompt += f"\nHere is the previous code that had issues:\n```python\n{current_code}\n```"
        if feedback: prompt += f"\nHere is the error feedback from the last run:\n{feedback}\nPlease fix the code."
        prompt += "\nIMPORTANT: Only output the Python code inside a markdown block."
        
        response = model.generate_content(prompt)
        current_code = _extract_python_code(response.text)
        all_passed, feedback = _execute_and_test_code(current_code, io_examples)
        
        print(f"Test Result: {feedback}")
        if all_passed:
            print(f"SUCCESS for {issue_key} on attempt {attempt + 1}!")
            solution_path = os.path.join('jira_attachments', issue_key, 'solution.py')
            os.makedirs(os.path.dirname(solution_path), exist_ok=True)
            with open(solution_path, 'w') as f: f.write(current_code)
            upload_attachment(issue_key, solution_path)
            return f"Successfully generated and verified Python solution for {issue_key}. 'solution.py' has been attached."

    return f"Failed to generate a working Python solution for {issue_key} after {max_retries} attempts. Last feedback: {feedback}"

# ==============================================================================
# === SQL STORY LOGIC (SIMPLIFIED FOR CONVERSION ONLY) =========================
# ==============================================================================

def _clean_llm_sql_response(raw_text: str) -> str:
    """Cleans the raw SQL response from LLM."""
    cleaned = raw_text.strip()
    if cleaned.startswith("```sql"): cleaned = cleaned[len("```sql"):].strip()
    elif cleaned.startswith("```"): cleaned = cleaned[len("```"):].strip()
    if cleaned.endswith("```"): cleaned = cleaned[:-len("```")].strip()
    if cleaned.endswith(";"): cleaned = cleaned[:-1].strip()
    return cleaned

def process_single_sql_story(issue_key: str, config: dict, model: GenerativeModel) -> str:
    """
    (SIMPLIFIED) Processes a SQL story: converts Oracle to PG and uploads the result.
    DATABASE EXECUTION IS SKIPPED.
    """
    print(f"Starting SQL Conversion ONLY process for {issue_key}...")
    base_path = os.path.join('jira_attachments', issue_key)
    output_sql_path = os.path.join(base_path, 'output_sql.sql')
    os.makedirs(base_path, exist_ok=True)

    try:
        # 1. Download attachments
        downloaded_files = download_attachments(issue_key)
        input_csv_path = next((f for f in downloaded_files if f.endswith('input_sql.csv')), None)
        
        if not input_csv_path:
            return f"Error: Could not find 'input_sql.csv' attached to story {issue_key}."
        
        print(f"Found input file: {input_csv_path}")
        input_df = pd.read_csv(input_csv_path)
        
        # 2. Convert Oracle to PostgreSQL
        with open(output_sql_path, 'w', encoding='utf-8') as f_out:
            for index, row in input_df.iterrows():
                oracle_query = str(row.get('oracle', '')).strip()
                if not oracle_query: continue
                
                prompt = (f"Convert the following Oracle SQL to a precise PostgreSQL equivalent.\n"
                          f"Oracle SQL:\n```sql\n{oracle_query}\n```\n"
                          "Respond ONLY with the converted PostgreSQL query. No explanations or markdown.")
                
                response = model.generate_content(prompt)
                pg_query = _clean_llm_sql_response(response.text)
                
                f_out.write(f"-- Converted query for original CSV row {index}\n")
                f_out.write(pg_query + "\n\n")

        print(f"Conversion complete. Generated SQL file: {output_sql_path}")
        
        # 3. Upload the result file
        print("Uploading converted SQL file to Jira...")
        upload_attachment(issue_key, output_sql_path)
        
        # 4. Return a success message
        return (f"Successfully converted Oracle to PostgreSQL for story {issue_key}. "
                f"The resulting file 'output_sql.sql' has been attached to the Jira ticket. "
                "Execution was not performed.")

    except Exception as e:
        print(f"A critical error occurred during conversion: {e}")
        # Attempt to upload the file even if an error occurred mid-process
        if os.path.exists(output_sql_path):
            upload_attachment(issue_key, output_sql_path)
        return (f"I am sorry, but an error occurred during the SQL conversion for {issue_key}: {e}. "
                "I have attempted to upload the partially converted file for your review.")
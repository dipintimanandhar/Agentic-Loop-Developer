# adk_developer_agent/app.py

import gradio as gr
import os
import logging
import sys
import time
import threading
from typing import Any, Dict, List
from uuid import UUID

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# LangChain / ADK imports
import vertexai
from langchain_google_vertexai import ChatVertexAI
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, AIMessage
from langchain.callbacks.base import BaseCallbackHandler

# Local module imports
import config_manager
from adk_developer_agent.tools.jira_tools import list_jira_stories, process_uploaded_notes_file
from adk_developer_agent.tools.developer_tools import process_sql_conversion_story, process_python_development_story

# --- Callback Handler to Capture Agent Thoughts ---
class AgentThoughtCallbackHandler(BaseCallbackHandler):
    def __init__(self):
        self.logs = []

    def on_agent_action(self, action: Any, **kwargs: Any) -> Any:
        tool_name = action.tool
        tool_input = action.tool_input
        log_message = f"🤔 **Thinking:** I need to use the tool `{tool_name}` with this input: `{tool_input}`."
        self.logs.append(log_message)
        print(log_message)

    def on_tool_end(self, output: Any, **kwargs: Any) -> Any:
        log_message = f"✅ **Observation:** The tool finished and returned: `{str(output)[:500]}...`."
        self.logs.append(log_message)
        print(log_message)

# --- Agent Initialization ---
# (This function remains unchanged)
def initialize_agent():
    # ... (code from previous version) ...
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    config_path = os.path.join(project_root, 'config.ini')
    config_params = config_manager.read_all_config(config_path)
    if not config_params:
        raise ValueError("Failed to load configuration from config.ini.")
    vertexai.init(project=config_params['project_id'], location=config_params['vertex_ai_location'])
    llm = ChatVertexAI(
        model_name=config_params['vertex_ai_model_name'], temperature=0, streaming=True,
        model_kwargs={"convert_system_message_to_human": True}
    )
    tools = [list_jira_stories, process_uploaded_notes_file, process_sql_conversion_story, process_python_development_story]
    prompt = ChatPromptTemplate.from_messages([
        ("system", (
            "You are a helpful and highly capable AI developer assistant. "
            "Your name is 'Loop'. When a user asks you to perform a task, you must choose the best tool to accomplish it. "
            "If the user uploads a file, your primary goal is to process it using the 'process_uploaded_notes_file' tool. The file path will be provided in the input. "
            "Always be clear and concise in your final response."
        )),
        ("placeholder", "{chat_history}"), ("human", "{input}"), ("placeholder", "{agent_scratchpad}"),
    ])
    agent = create_tool_calling_agent(llm, tools, prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, return_intermediate_steps=True)
    return agent_executor
# ... (end of unchanged function) ...

agent_executor = initialize_agent()

# --- Gradio UI Logic with Live Animation ---
def handle_agent_interaction(message, history, file_upload):
    
    # --- Part 1: Prepare the agent's task ---
    chat_history_langchain_format = []
    for human, ai in history:
        chat_history_langchain_format.append(HumanMessage(content=human))
        chat_history_langchain_format.append(AIMessage(content=ai))
    
    thought_handler = AgentThoughtCallbackHandler()

    if file_upload is not None:
        agent_input = f"A user has uploaded a file named '{os.path.basename(file_upload.name)}'. Process it. Path: {file_upload.name}"
        message_to_show = f"Processing uploaded file: `{os.path.basename(file_upload.name)}`"
    else:
        if not message.strip():
            return history, "", None, "No input provided.", ""
        agent_input = message
        message_to_show = message

    # --- Part 2: Instantly update UI and start the agent in a background thread ---
    history.append((message_to_show, ""))
    yield history, "", None, "🤔 Agent is thinking...", ""

    result_container = {"result": None} # A container to get the result from the thread

    def agent_worker():
        """The function that the background thread will run."""
        response = agent_executor.invoke(
            {"input": agent_input, "chat_history": chat_history_langchain_format},
            {"callbacks": [thought_handler]}
        )
        result_container["result"] = response

    agent_thread = threading.Thread(target=agent_worker)
    agent_thread.start()

    # --- Part 3: Run the animation loop while the agent works ---
    animation_chars = ["...", "·..", ".·.", "..·"]
    i = 0
    while agent_thread.is_alive():
        # Update the last bot message with the next animation frame
        thinking_message = f"{animation_chars[i % len(animation_chars)]}"
        history[-1] = (message_to_show, thinking_message)
        yield history, "", None, "🤔 Agent is thinking...", ""
        time.sleep(0.5)
        i += 1
    
    # --- Part 4: The agent is done. Get the result and show the final output. ---
    response = result_container["result"]
    thought_process_md = "### Agent's Internal Monologue\n\n" + "\n\n".join(thought_handler.logs)
    history[-1] = (message_to_show, response['output'])
    
    yield history, "", None, "✅ Done! Waiting for your next command.", thought_process_md

# --- Build and Launch the Gradio Interface ---
# (This part remains unchanged)
script_dir = os.path.dirname(__file__)
agent_logo_path = os.path.join(script_dir, "assets", "agent_logo.png")
user_avatar_path = os.path.join(script_dir, "assets", "user_avatar.png")

with gr.Blocks(theme=gr.themes.Soft(primary_hue="blue", secondary_hue="sky"), title="Loop Developer Agent") as demo:
    # ... (UI layout code from previous version) ...
    with gr.Row():
        gr.Image(agent_logo_path, width=100, scale=0, container=False, show_download_button=False)
        with gr.Column(scale=4):
            gr.Markdown("# Loop: Your AI Developer Agent")
            gr.Markdown("I can create and process Jira stories, convert code, and more. Give me a task or upload meeting notes to get started.")
    with gr.Row():
        with gr.Column(scale=2):
            chatbot = gr.Chatbot(label="Conversation", height=500, bubble_full_width=False, avatar_images=(user_avatar_path, agent_logo_path))
            status_display = gr.Markdown("*Waiting for your command...*")
            with gr.Row():
                msg = gr.Textbox(
                    label="Your Message", placeholder="e.g., 'list all jira stories' or 'complete user story KAN-30'",
                    scale=4,
                )
                file_upload = gr.File(label="Upload Notes")
            gr.Examples(
                examples=["list all jira stories", "Please process the sql conversion for story KAN-30", "can you complete the python development for story KAN-31"],
                inputs=msg, label="Example Prompts"
            )
        with gr.Column(scale=1):
            gr.Markdown("## Agent Internals")
            agent_thoughts = gr.Markdown("*The agent's thought process will appear here...*")
    outputs = [chatbot, msg, file_upload, status_display, agent_thoughts]
    msg.submit(handle_agent_interaction, [msg, chatbot, file_upload], outputs, queue=True)
    file_upload.upload(handle_agent_interaction, [msg, chatbot, file_upload], outputs, queue=True)
# ... (end of unchanged UI layout) ...

if __name__ == "__main__":
    print("Launching AI Developer Agent UI...")
    demo.launch(debug=True)
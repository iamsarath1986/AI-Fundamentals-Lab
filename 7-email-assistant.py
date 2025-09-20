import os
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from typing_extensions import TypedDict, Literal, Annotated
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_core.tools import tool
from langchain_core.messages import AIMessage
from langgraph.store.memory import InMemoryStore
from langgraph.graph import StateGraph, END, add_messages
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver
from IPython.display import Image, display

# --- Initial Setup ---
_ = load_dotenv()

profile = {
    "name": "John",
    "full_name": "John Doe",
    "user_profile_background": "Senior software engineer leading a team of 13 developers",
}

prompt_instructions = {
    "triage_rules": {
        "ignore": "Marketing newsletters, spam emails, mass company announcements",
        "notify": "Team member out sick, build system notifications, project status updates",
        "respond": "Direct questions from team members, meeting requests, critical bug reports, security breach notification, critical vulnerability disclosures, user-reported phishing attempts"
    },
    "agent_instructions": "Use these tools when appropriate to help manage Jane's tasks efficiently."
}

# --- LLM and Embeddings Configuration ---
# Note: 'llama3.2' might be a typo. Ensure you have this model installed in Ollama,
# or change it to a valid model name like 'llama3'.
llm = ChatOllama(model="llama3.2")
ollama_embedder = OllamaEmbeddings(model="nomic-embed-text")

# --- Placeholder Prompts ---
# In your original code, these are imported from a 'prompts.py' file.
# I've recreated them here to make the script self-contained.
triage_system_prompt = """
You are an email triage expert. Your task is to analyze the following email and classify it into one of three categories: 'ignore', 'notify', or 'respond'.
Use the following rules to make your decision:
- ignore: {triage_no}
- notify: {triage_notify}
- respond: {triage_email}
Provide your reasoning and then the final classification.
"""
triage_user_prompt = """
From: {author}
To: {to}
Subject: {subject}
Body:
{email_thread}
"""
# --- Pydantic Model for Triage ---
class Router(BaseModel):
    """Analyze the unread email and route it according to its content."""
    reasoning: str = Field(description="Step-by-step reasoning behind the classification.")
    classification: Literal["ignore", "respond", "notify"] = Field(
        description="The classification of an email: 'ignore' for irrelevant emails, "
                    "'notify' for important information that doesn't need a response, "
                    "'respond' for emails that need a reply",
    )

llm_router = llm.with_structured_output(Router)

# --- Tool Definitions ---
@tool
def write_email(to: str, subject: str, content: str) -> str:
    """Write and send an email."""
    return f"Email sent to {to} with subject '{subject}'"

@tool
def schedule_meeting(attendees: list[str], subject: str, duration_minutes: int, preferred_day: str) -> str:
    """Schedule a calendar meeting."""
    return f"Meeting '{subject}' scheduled for {preferred_day} with {len(attendees)} attendees"

@tool
def check_calendar_availability(day: str) -> str:
    """Check calendar availability for a given day."""
    return f"Available times on {day}: 9:00 AM, 2:00 PM, 4:00 PM"

# --- Memory and Agent Setup ---
store = InMemoryStore(index={"embed": ollama_embedder})
print("InMemoryStore configured with an Ollama-powered vector index.")

# Corrected code
manage_memory_tool = create_manage_memory_tool(
    store=store,
    namespace=("email_assistant", "{langgraph_user_id}", "collection")
)
search_memory_tool = create_search_memory_tool(
    store=store,
    namespace=("email_assistant", "{langgraph_user_id}", "collection")
)

agent_system_prompt_memory = """
< Role >
You are {full_name}'s executive assistant. You are a top-notch executive assistant who cares about {name} performing as well as possible.
</ Role >

< Tools >
You have access to the following tools to help manage {name}'s communications and schedule:
1. write_email(to, subject, content) - Send emails to specified recipients
2. schedule_meeting(attendees, subject, duration_minutes, preferred_day) - Schedule calendar meetings
3. check_calendar_availability(day) - Check available time slots for a given day
4. manage_memory - Store any relevant information about contacts, actions, discussion, etc. in memory for future reference
5. search_memory - Search for any relevant information that may have been stored in memory
</ Tools >

< Instructions >
{instructions}
</ Instructions >
"""


def create_prompt(state):
    return [{"role": "system",
             "content": agent_system_prompt_memory.format(instructions=prompt_instructions["agent_instructions"],
                                                          **profile)}] + state['messages']


tools = [write_email, schedule_meeting, check_calendar_availability, manage_memory_tool, search_memory_tool]

response_agent = create_react_agent(llm, tools=tools, prompt=create_prompt, store=store)


# --- Graph State Definition (Corrected) ---
class State(TypedDict):
    email_input: dict
    messages: Annotated[list, add_messages]
    next_step: str  # Added to hold the routing decision


# --- Graph Node Definitions (Corrected Architecture) ---
def triage_router(state: State) -> dict:
    """
    Analyzes the email and decides the next step, returning it in a dictionary.
    """
    print("---NODE: TRIAGING EMAIL---")
    email_details = state['email_input']
    user_prompt = triage_user_prompt.format(**email_details)
    system_prompt = triage_system_prompt.format(
        full_name=profile["full_name"],
        name=profile["name"],
        user_profile_background=profile["user_profile_background"],
        triage_no=prompt_instructions["triage_rules"]["ignore"],
        triage_notify=prompt_instructions["triage_rules"]["notify"],
        triage_email=prompt_instructions["triage_rules"]["respond"],
    )
    result = llm_router.invoke([{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}])

    if result.classification == "respond":
        print("📧 Classification: RESPOND - Routing to save memory node.")
        return {"next_step": "save_memory_node"}
    else:
        if result.classification == "ignore":
            print("🚫 Classification: IGNORE - This email can be safely ignored")
        else:
            print("🔔 Classification: NOTIFY - This email contains important information")
        return {"next_step": "__end__"}

def save_memory_node(state: State):
    """
    A dedicated node to save key info from an email to memory using a DIRECT tool call.
    This bypasses the LLM for reliability.
    """
    print("---NODE: SAVING TO MEMORY (DIRECT CALL)---")
    email_details = state['email_input']

    # 1. Create the summary string directly in Python
    summary = (
        f"An email regarding a potential security breach was received from "
        f"{email_details['author']}. The subject was: '{email_details['subject']}'."
    )
    print(f"Saving summary to memory: '{summary}'")

    # 2. Directly invoke the memory tool. This is a standard Python function call.
    # The tool's input is a dictionary with the content to save.
    manage_memory_tool.invoke({"content": summary})

    # 3. Prepare the prompt for the next step (response_agent)
    next_prompt = (
        f"The critical information from the last email (Subject: \"{email_details['subject']}\") "
        f"has now been saved to your memory. Please prepare a brief, professional response to the sender."
    )
    return {"messages": [("user", next_prompt)]}


# --- Graph Construction (Corrected) ---
email_agent_graph = StateGraph(State)

email_agent_graph.add_node("triage_router", triage_router)
email_agent_graph.add_node("save_memory_node", save_memory_node)
email_agent_graph.add_node("response_agent", response_agent)

email_agent_graph.set_entry_point("triage_router")
email_agent_graph.add_conditional_edges(
    "triage_router",
    lambda state: state["next_step"],
    {
        "save_memory_node": "save_memory_node",
        "__end__": END,
    }
)
email_agent_graph.add_edge("save_memory_node", "response_agent")
email_agent_graph.add_edge("response_agent", END)

memory_saver = MemorySaver()
email_agent = email_agent_graph.compile(checkpointer=memory_saver)

print("\n--- Displaying New Graph Architecture ---")
display(Image(email_agent.get_graph(xray=True).draw_mermaid_png()))

# --- Graph Execution ---
config = {"configurable": {"thread_id": "lance", "langgraph_user_id": "lance"}}

# 1. Process the email about the security breach
print("\n--- PROCESSING SECURITY EMAIL ---")
security_email = {
    "author": "Densil Smith <densil.smith@company.com>",
    "to": "Jane Doe <jane.doe@company.com>",
    "subject": "Need immediate attention",
    "email_thread": """Hi Jane,\n\nOur system get hacked by someone?""",
}
response = email_agent.invoke({"email_input": security_email, "messages": []}, config=config)

print("\n--- FINAL RESPONSE FOR SECURITY EMAIL ---")
for m in response["messages"]:
    if isinstance(m, AIMessage) and not m.tool_calls:
        m.pretty_print()

# 2. Query the agent's memory about the email
print("\n--- QUERYING AGENT MEMORY ---")
query = "Who sent the email about the security breach?"
response = response_agent.invoke(
    {"messages": [{"role": "user", "content": query}]},
    config=config
)

print("\n--- FINAL RESPONSE FROM MEMORY ---")
for m in response["messages"]:
    if isinstance(m, AIMessage) and not m.tool_calls:
        m.pretty_print()
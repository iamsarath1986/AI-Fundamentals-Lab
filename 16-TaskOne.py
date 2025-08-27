"""
Task:
We spend too much time drafting documents and this need to be fixed!
For the company, we need to create an AI Agentic System that can speed up drafting documents, emails, etc. The AI
Agentic System should have Human-AI Collaboration meaning the Human should be able to provide continuous feedback and the AI Agent should stop when the Human is happy with the draft. The system should be able to save the drafts fast.
"""
from typing import Annotated, Sequence, TypedDict
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage, SystemMessage
from langchain_ollama import ChatOllama
from langchain.tools import tool
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode

document_content = ""

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]

@tool
def update(content: str) -> str:
    """Updates the document with the provided content."""
    global document_content
    document_content = content
    return f"Document has been updated successfully! The current content is: \n{document_content}"

@tool
def save(filename: str) -> str:
    """Saves the document to a text file and finish the process.

    Args:
        filename: Name for the text file.
    """

    if not filename.endswith(".txt"):
        filename = filename + ".txt"

    try:
        with open(filename, "w") as file:
            file.write(document_content)
        return f"Document has been saved to: {filename}"

    except Exception as e:
        return f"Error saving document: {str(e)}"

tools = [update, save]

model = ChatOllama(model="mistral").bind_tools(tools)

def agent(state: AgentState) -> AgentState:
    system_prompt = SystemMessage(content=f"""
    You are Drafter, a helpful writing assistant. You are going to help the user update and modify documents.
    
    - If the user wants to update or modify content, use the 'update' tool with the complete update content.
    - If the user wants to save and finish, you need to use the 'save' tool.
    - Make sure to always show the current document state after modifications.
    
    The current document state is: {document_content}
    """)

    if not state["messages"]:
        user_input = input("\nI am ready to help you update a document. What would you like to create?\n")
        user_message = HumanMessage(content=user_input)
    else:
        user_input = input("\nWhat would you like to do with the document?\n")
        user_message = HumanMessage(content=user_input)

    all_messages = [system_prompt] + list(state["messages"]) + [user_message]

    response = model.invoke(all_messages)

    print(f"\nAI: {response.content}")

    return {"messages": list(state["messages"]) + [user_message, response]}

def should_continue(state: AgentState) -> str:
    """Determine if we should continue or end the conversation"""

    messages = state["messages"]

    if not messages:
        return "continue"

    for message in reversed(messages):
        if (isinstance(message, ToolMessage) and
            "saved" in message.content.lower() and
            "document" in message.content.lower()):
            return "end"

    return "continue"

def print_messages(messages) -> None:
    """Functions I made to print the messages in a more readable format."""
    if not messages:
        return

    for message in messages:
        if isinstance(message, ToolMessage):
            print(f"\nTool Result: {message.content}")

graph = StateGraph(AgentState)
graph.add_node("agent", agent)
graph.add_node("tools", ToolNode(tools))
graph.add_edge(START, "agent")
graph.add_edge("agent", "tools")
graph.add_conditional_edges(
    "tools",
    should_continue,
    {
        "continue": "agent",
        "end": END,
    }
)

app = graph.compile()

def run_document_agent():
    print("\n=======Drafters=======")

    state = {"messages": []}

    for step in app.stream(state, stream_mode="values"):
        if "messages" in step:
            print_messages(step["messages"])

    print("\n=======Drafter Finished=======")

if __name__ == "__main__":
    run_document_agent()
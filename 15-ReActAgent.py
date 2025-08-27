from typing import Annotated, Sequence, TypedDict
from langchain_core.messages import BaseMessage, ToolMessage, SystemMessage
from langchain_ollama import ChatOllama
from langchain.tools import tool
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode
from dotenv import load_dotenv

#BaseMessage: Foundational message types in Langgraph
#ToolMessage: Pass data back to LLM after it calls a tool
#SystemMessage: Message for providing instructions to LLM

load_dotenv()

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]

@tool
def addition(a: int, b: int) -> int:
    """This is an addition function that adds two numbers."""
    return a + b

@tool
def subtract(a: int, b: int) -> int:
    """This is a subtraction function that subtracts two numbers."""
    return a - b

@tool
def multiply(a: int, b: int) -> int:
    """This is a multiplication function that multiply two numbers."""
    return a * b

@tool
def division(a: int, b: int) -> int:
    """This is a division function that divide two numbers."""
    return a / b

tools = [addition, subtract, multiply, division]

model = ChatOllama(model="mistral").bind_tools(tools)

def model_call(state:AgentState) -> AgentState:
    response = model.invoke(state["messages"])
    return {"messages": [response]}

def should_continue(state: AgentState):
    messages = state["messages"]
    last_message = messages[-1]
    if not last_message.tool_calls:
        return "end"
    else:
        return "continue"

graph = StateGraph(AgentState)
graph.add_node("our_agent", model_call)

tool_node = ToolNode(tools=tools)
graph.add_node("tools", tool_node)

graph.set_entry_point("our_agent")
graph.add_conditional_edges(
    "our_agent",
    should_continue,
    {
        "continue": "tools",
        "end": END,
    }
)

graph.add_edge("tools", "our_agent")

app = graph.compile()

def print_stream(stream):
    for s in stream:
        s["messages"][-1].pretty_print()

inputs = {
    "messages": [
        SystemMessage(content="You are my AI assistant, please answer my query to the best of your ability."),
        ("user", "Add 300 + 700 and then divide the result by 2")
    ]
}
print_stream(app.stream(inputs, stream_mode="values"))
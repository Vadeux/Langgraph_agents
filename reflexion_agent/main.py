from typing import Any, Literal

from chains import first_responder_chain, revisor_chain
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.constants import START
from langgraph.graph import END, MessagesState, StateGraph
from tool_executor import execute_tools

MAX_ITERATIONS = 5


def draft_node(state: MessagesState) -> dict[str, Any]:
    """Generate a draft initial response."""
    draft = first_responder_chain.invoke(state["messages"])
    return {"messages": [draft]}


def revise_node(state: MessagesState) -> dict[str, Any]:
    """Revise the draft response based on the tool response."""
    revised_response = revisor_chain.invoke(state["messages"])
    return {"messages": [revised_response]}


def event_tool(state: MessagesState) -> Literal["execute_tools", END]:
    """Determine whether to continue or end based on the iteration counter."""
    if (
        sum(isinstance(item, ToolMessage) for item in state["messages"])
        > MAX_ITERATIONS
    ):
        return END
    return "execute_tools"


flow = StateGraph(state_schema=MessagesState)
flow.add_node("draft", draft_node)
flow.add_node("execute_tools", execute_tools)
flow.add_node("revise", revise_node)

flow.add_edge(START, "draft")

flow.add_edge("draft", "execute_tools")
flow.add_edge("execute_tools", "revise")
flow.add_conditional_edges(
    "revise", event_tool, path_map={END: END, "execute_tools": "execute_tools"}
)

graph = flow.compile()

if __name__ == "__main__":
    res = graph.invoke(
        {
            "messages": [
                HumanMessage(
                    content="write about top-5 professional dota2 teams from 2025 to 2026?"
                )
            ]
        }
    )
    last_message = res["messages"][-1]
    if isinstance(last_message, AIMessage) and last_message.tool_calls:
        print(last_message.tool_calls[0]["args"]["answer"])
    print(res)

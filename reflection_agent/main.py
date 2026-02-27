from typing import Annotated, Any

from dotenv import load_dotenv
from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.graph import END, MessagesState, StateGraph
from langgraph.graph.message import add_messages

load_dotenv()
from chains import generation_chain, reflection_chain


class CustomGraphState(MessagesState):
    """Redefine custom messages field to support `chat_history` in MessagesPlaceholder"""

    chat_history: Annotated[list[BaseMessage], add_messages]


REFLECT = "reflect"
GENERATE = "generate"
MAX_ITERATIONS = 5


def generate_node(state: CustomGraphState) -> dict[str, Any]:
    """Generate a tweet based on a user request and critic."""
    gen_ai_response = generation_chain.invoke({"chat_history": state["chat_history"]})
    return {"chat_history": [gen_ai_response]}  # Why []? because `add_messages`


def reflect_node(state: CustomGraphState) -> dict[str, Any]:
    """Generate a critique and recommendations for the user's tweet."""
    reflection_result = reflection_chain.invoke({"chat_history": state["chat_history"]})
    return {
        "chat_history": [HumanMessage(content=reflection_result.content)]
    }  # we wrap it with HumanMessage to make it more valuable for the LLM


def choose_next_node(state: CustomGraphState) -> str:
    if len(state["chat_history"]) > MAX_ITERATIONS:
        return END
    return REFLECT


flow = StateGraph(state_schema=CustomGraphState)

flow.add_node(GENERATE, generate_node)
flow.add_node(REFLECT, reflect_node)

flow.set_entry_point(GENERATE)

flow.add_conditional_edges(
    GENERATE, choose_next_node, path_map={END: END, REFLECT: REFLECT}
)
flow.add_edge(REFLECT, GENERATE)

graph = flow.compile()
graph.get_graph().draw_mermaid_png(output_file_path="reflection_agent/flow_diagram.png")

if __name__ == "__main__":
    print("Start reflection agent.")
    inputs = HumanMessage(content="""
            Make this tweet better:
            "
                Announcing The International 2022 Swag Bag for all active Dota customers --
                celebrate a joyous season of Dota by claiming your free Arcana, Battle Pass, and access to Dota Plus.
                https://dota2.com/newsentry/3398555399418412272
            "
        """)
    result = graph.invoke({"chat_history": inputs})
    print(result["chat_history"][-1].content)

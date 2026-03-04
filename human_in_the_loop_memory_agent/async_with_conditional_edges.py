import operator
from typing import Annotated, Any, Sequence, TypedDict

from dotenv import load_dotenv
from langgraph.graph import END, START, StateGraph

load_dotenv()


class State(TypedDict):
    aggregate: Annotated[list[str], operator.add]
    which: list[str]


class ReturnNodeValue:
    def __init__(self, node_secret: str) -> None:
        self._value = node_secret

    def __call__(self, state: State) -> Any:
        import time

        time.sleep(1)
        print(f"Adding {self._value} to state: {state['aggregate']}")
        return {"aggregate": [self._value]}


def route_bc_or_cd(state: State) -> Sequence[str]:
    if sorted(state["which"]) == sorted(["c", "d"]):
        return ["c", "d"]
    return ["b", "c"]


intermediates = ["b", "c", "d"]

flow = StateGraph(State)
flow.add_node("a", ReturnNodeValue("I am A"))
flow.add_node("b", ReturnNodeValue("I am B"))
flow.add_node("c", ReturnNodeValue("I am C"))
flow.add_node("d", ReturnNodeValue("I am D"))
flow.add_node("e", ReturnNodeValue("I am E"))

flow.add_edge(START, "a")
flow.add_conditional_edges("a", route_bc_or_cd, intermediates)

for node in intermediates:
    flow.add_edge(node, "e")
flow.add_edge("e", END)


graph = flow.compile()
graph.get_graph().draw_mermaid_png(
    output_file_path="flow_diagram_async_conditional.png"
)


if __name__ == "__main__":
    print("Hello async conditional!")
    graph.invoke(
        {"aggregate": [], "which": ["c", "d"]}, {"configurable": {"thread_id": "buz"}}
    )

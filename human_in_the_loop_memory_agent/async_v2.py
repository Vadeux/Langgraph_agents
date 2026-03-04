import operator
from typing import Annotated, Any, TypedDict

from dotenv import load_dotenv
from langgraph.graph import END, START, StateGraph

load_dotenv()


class State(TypedDict):
    aggregate: Annotated[list[str], operator.add]


class ReturnNodeValue:
    def __init__(self, node_secret: str) -> None:
        self._value = node_secret

    def __call__(self, state: State) -> Any:
        import time

        time.sleep(1)
        print(f"Adding {self._value} to state: {state['aggregate']}")
        return {"aggregate": [self._value]}


flow = StateGraph(State)
flow.add_node("a", ReturnNodeValue("I am A"))
flow.add_node("b", ReturnNodeValue("I am B"))
flow.add_node("b2", ReturnNodeValue("I am B2"))
flow.add_node("c", ReturnNodeValue("I am C"))
flow.add_node("d", ReturnNodeValue("I am D"))

flow.add_edge(START, "a")
flow.add_edge("a", "b")
flow.add_edge("a", "c")
flow.add_edge("b", "b2")
flow.add_edge(["b2", "c"], "d")  # creates 2 edges
flow.add_edge("d", END)

graph = flow.compile()
graph.get_graph().draw_mermaid_png(output_file_path="flow_diagram_async_v2.png")


if __name__ == "__main__":
    print("Hello async2!")
    graph.invoke({"aggregate": []}, {"configurable": {"thread_id": "boo"}})

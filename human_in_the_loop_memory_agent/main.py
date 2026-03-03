from typing import TypedDict

from dotenv import load_dotenv
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

load_dotenv()


class State(TypedDict):
    input: str
    user_feedback: str


def step_1(state: State) -> None:
    print("---Step 1---")


def human_feedback(state: State) -> None:
    print("---Human feedback---")
    print(state["user_feedback"])


def step_3(state: State) -> None:
    print("---Step 3---")


flow = StateGraph(State)
flow.add_node("step_1", step_1)
flow.add_node("human_feedback", human_feedback)
flow.add_node("step_3", step_3)

flow.add_edge(START, "step_1")

flow.add_edge("step_1", "human_feedback")
flow.add_edge("human_feedback", "step_3")

flow.add_edge("step_3", END)

memory = MemorySaver()

graph = flow.compile(checkpointer=memory, interrupt_before=["human_feedback"])
graph.get_graph().draw_mermaid_png(output_file_path="flow_diagram.png")

if __name__ == "__main__":
    thread = {"configurable": {"thread_id": 1}}

    initial_input = {"input": "hello world"}

    for event in graph.stream(initial_input, thread, stream_mode="values"):
        print(event)

    print(graph.get_state(thread).next)

    user_input = input("Tell me how you want to update the state: ")

    graph.update_state(thread, {"user_feedback": user_input}, as_node="human_feedback")

    print("--State after update--")
    print(graph.get_state(thread))

    print(graph.get_state(thread).next)

    for event in graph.stream(None, thread, stream_mode="values"):
        print(event)

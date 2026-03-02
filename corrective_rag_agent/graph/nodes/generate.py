from typing import Any

from corrective_rag_agent.graph.chains.generation import generation_chain
from corrective_rag_agent.graph.state import GraphState


def generate(state: GraphState) -> dict[str, Any]:
    print("---GENERATE---")
    question = state["question"]
    documents = state["documents"]
    generated_answer = generation_chain.invoke(
        {"question": question, "context": documents}
    )
    return {"generation": generated_answer}

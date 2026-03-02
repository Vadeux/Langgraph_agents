from typing import Any

from corrective_rag_agent.graph.state import GraphState
from corrective_rag_agent.ingestion import retriever


def retrieve(state: GraphState) -> dict[str, Any]:
    print("---RETRIEVE---")
    question = state["question"]
    documents = retriever.invoke(question)
    return {"documents": documents}

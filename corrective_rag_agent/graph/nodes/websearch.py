from typing import Any

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_tavily import TavilySearch

from corrective_rag_agent.graph.state import GraphState

load_dotenv()

web_search_tool = TavilySearch(max_results=3)


def web_search(state: GraphState) -> dict[str, Any]:
    print("---WEB SEARCH---")
    question = state["question"]
    documents = state.get("documents", []) if "documents" in state else None
    tavily_results = web_search_tool.invoke(question)["results"]
    joined_tavily_result = "\n".join(
        [tavily_doc["content"] for tavily_doc in tavily_results]
    )
    web_results = Document(page_content=joined_tavily_result)
    if documents is not None:
        documents.append(web_results)
    else:
        documents = [web_results]
    return {"documents": documents}

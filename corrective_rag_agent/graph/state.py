import operator
from typing import Annotated, TypedDict

from langchain_core.documents import Document


class GraphState(TypedDict):
    """
    Represents the state of the graph.

    Attributes:
        question: question (initial query)
        generation: LLM generation (answer of the llm)
        web_search: whether to add search (bool, to indicate that we need to search for smth)
        documents: list of documents (relevant which help us to answer the question)
    """

    question: str
    generation: str
    web_search: bool
    documents: Annotated[list[Document], operator.add]  # list[str]

from pprint import pprint

from dotenv import load_dotenv

from corrective_rag_agent.graph.chains.generation import generation_chain
from corrective_rag_agent.graph.chains.hallucination_grader import (
    GradeHallucinations,
    hallucination_grader,
)

load_dotenv()
from corrective_rag_agent.graph.chains.retrieval_grader import (
    GradeDocuments,
    retrieval_grader_chain,
)
from corrective_rag_agent.ingestion import retriever


def test_retrieval_grader_answer_yes() -> None:
    question = "agent memory"
    docs = retriever.invoke(question)
    doc_txt = docs[0].page_content

    result: GradeDocuments = retrieval_grader_chain.invoke(
        {"question": question, "document": doc_txt}
    )

    assert result.binary_score == "yes"


def test_retrieval_grader_answer_no() -> None:
    question = "agent memory"
    docs = retriever.invoke(question)
    doc_txt = docs[0].page_content

    result: GradeDocuments = retrieval_grader_chain.invoke(
        {"question": "how to make pizza", "document": doc_txt}
    )
    assert result.binary_score == "no"


def test_generation_chain() -> None:
    question = "agent memory"
    docs = retriever.invoke(question)
    generation = generation_chain.invoke({"question": question, "context": docs})
    pprint(generation)


def test_hallucination_grader_answer_yes() -> None:
    question = "agent memory"
    docs = retriever.invoke(question)

    generation = generation_chain.invoke({"question": question, "context": docs})
    res: GradeHallucinations = hallucination_grader.invoke(
        {"documents": docs, "generation": generation}
    )
    assert res.binary_score


def test_hallucination_grader_answer_no() -> None:
    question = "agent memory"
    docs = retriever.invoke(question)

    res: GradeHallucinations = hallucination_grader.invoke(
        {
            "documents": docs,
            "generation": "In order to make pizza we need to start with the dog",
        }
    )
    assert not res.binary_score

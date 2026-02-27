from pydantic import BaseModel, Field


class Reflection(BaseModel):
    missing: str = Field(description="Critique of what is missing.")
    useless_information: str = Field(description="Critique of what is not useful.")
    search_queries: list[str] = Field(
        description="1-3 search queries for researching improvements to address the critique of your current answer."
    )


class AnswerQuestion(BaseModel):
    answer: str = Field(description="~250 words detailed answer to the question.")
    reflection: Reflection = Field(description="Reflection on the initial answer.")


class ReviseAnswer(AnswerQuestion):
    """Revise your original answer to the question."""

    references: list[str] = Field(description="List of references used in your answer.")

import datetime

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langchain_core.output_parsers.openai_tools import JsonOutputToolsParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from schemas import AnswerQuestion, ReviseAnswer

load_dotenv()


llm = ChatOpenAI(model="gpt-4o")
parser = JsonOutputToolsParser(return_id=True)

actor_prompt_template = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
                You are an expert researcher.
                Current time: {current_time}
                
                1. {first_instructions}
                2. Reflect and critique your answer. Be serve to maximize improvement.
                3. Recommend search queries to research information and improve your answer.
            """,
        ),
        MessagesPlaceholder("messages"),
        ("system", "Answer the user's question above using the required format."),
    ]
).partial(current_time=datetime.datetime.now().isoformat())

first_responder_prompt_template = actor_prompt_template.partial(
    first_instructions="Provide a detailed ~250 word answer."
)

first_responder_chain = first_responder_prompt_template | llm.bind_tools(
    tools=[AnswerQuestion], tool_choice="AnswerQuestion"
)  # we push llm to use tool `AnswerQuestion`

revise_instructions = """Revise your previous answer using the new information.
    - You should use the previous critique to add important information to your answer.
        - You MUST include numerical citations in your revised answer to ensure it can be verified.
        - Add a "References" section to the bottom of your answer (which does not count towards the word limit). In form of:
            - [1] https://example.com
            - [2] https://example.com
    - You should use the previous critique to remove superfluous information from your answer and make SURE it is not more than 250 words.
"""
revise_instructions_prompt_template = actor_prompt_template.partial(
    first_instructions=revise_instructions
)

revisor_chain = revise_instructions_prompt_template | llm.bind_tools(
    tools=[ReviseAnswer], tool_choice="ReviseAnswer"
)  # we push llm to use tool `ReviseAnswer`


if __name__ == "__main__":

    result = first_responder_chain.invoke(
        input={
            "messages": [
                HumanMessage(
                    content="write about top-5 professional dota2 teams in late 2025?"
                )
            ]
        }
    )
    print(result.answer)

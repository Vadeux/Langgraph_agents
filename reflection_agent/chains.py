from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI

llm = ChatOpenAI()

reflection_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
                You are a viral twitter influencer grading a tweet.
                Generate critique and recommendations for the user's tweet.
                Always provide detailed recommendations, including requests for length, virality, style, etc.
            """,
        ),
        MessagesPlaceholder(variable_name="chat_history"),
    ]
)

reflection_chain = reflection_prompt | llm

generation_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
                 You are a twitter techie influencer assistant tasked with writing excellent twitter posts.
                 Generate the best twitter post possible for the user's request.
                 If the user provides critique, respond with a revised version of your previous attempts.
                 Return only the text of the tweet, nothing else.
                 If the user provides no critique, respond with a new tweet.
             """,
        ),
        MessagesPlaceholder(variable_name="chat_history"),
    ]
)

generation_chain = generation_prompt | llm

import os
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI

load_dotenv()

reflection_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a viral twitter influencer grading a tweet. Generate critique and recommendations for the user's tweets."
            "Always provide detailed recommendations , including requests for length, virality, style, etc."
        ),
        MessagesPlaceholder(variable_name="messages"),
    ]
)

generation_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a techie twitter influencer assistant tasked with writing excellent twitter posts."
            "Generate the best twitte posts possible fot the user's request."
            "If the user provides critique, respond with a revised version of your previous attempts.",
        ),
        MessagesPlaceholder(variable_name="messages")
    ]
)

llm = ChatOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

generate_chain = generation_prompt | llm
reflect_chain = reflection_prompt | llm

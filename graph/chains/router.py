import os
from dotenv import load_dotenv
from typing import Literal

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from pydantic import BaseModel, Field

load_dotenv()


class RouteQuery(BaseModel):
    """Roue a user query to the most relevant datasource."""

    datasource: Literal["vectorstore", "websearch"] = Field(
        ...,
        description="Given a user description choose to route it to a websearch or vectorstore.",
    )


llm = ChatOpenAI(api_key=os.getenv("OPENAI_API_KEY"), temperature=0)
structured_llm_router = llm.with_structured_output(
    RouteQuery, method="function_calling")

system = """You are an expert at routing a user question to web search or vectorstore. \n
    The vectorstore contains documents related to LLM Powered Autonomous Agents, Reducing Toxicity in Language Models, Large Transformer Model Inference Optimization, short-term-memory, and Prompt Engineering. \n
    Use the vectorstore for questions on these topics. For all else, use web search."""

router_prompt_template = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "{question}")
    ]
)

question_router = router_prompt_template | structured_llm_router

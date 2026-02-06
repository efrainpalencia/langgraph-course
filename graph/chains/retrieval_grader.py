import os

from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

load_dotenv()

# Initialize LLM
llm = ChatOpenAI(api_key=os.getenv("OPENAI_API_KEY"), temperature=0)


class GradeDocuments(BaseModel):
    """Binary score for relevance check on retrieved documents."""

    binary_score: str = Field(
        description="Documents are relevant to the question, 'yes' or 'no'."
    )


# Structures output for the LLM with the given schema
structured_llm_grader = llm.with_structured_output(
    GradeDocuments, method="function_calling"
)

# System prompt
system = """You are a grader assessing relevance of a retrieved document of a user question. \n
        If the document conatins keyword(s) or semantic meaning related to the question, grade it as relevant. \n
        Give a binary score of "yes" or "no" score to indicate whether the document is relevant to the question."""

# # Prompt the LLM to retrieve the document from the question
grade_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "Retrieved document: \n\n{document} \n\n Question: \n\n{question}"),
    ]
)

# Chains using LCEL
retrieval_grader = grade_prompt | structured_llm_grader

import os

from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from langsmith import Client

load_dotenv()

# Initialize LLM
llm = ChatOpenAI(api_key=os.getenv("OPENAI_API_KEY"), temperature=0)

# Initialize Langchain Hub
client = Client(api_key=os.getenv("LANGSMITH_API_KEY"))

# Pulls prompt "rlm/rag-prompt" from Langchain Hub
prompt = client.pull_prompt("rlm/rag-prompt")

# Chains using LCEL
generation_chain = prompt | llm | StrOutputParser()

from dotenv import load_dotenv

from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch

load_dotenv()


@tool
def triple(num: float) -> float:
    """
    Docstring for triple

    :param num: A nu,ber to triple
    :type num: float
    :return: The triple of the input number
    :rtype: float
    """

    return float(num) * 3


tools = [TavilySearch(max_results=1), triple]

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.1).bind_tools(tools)

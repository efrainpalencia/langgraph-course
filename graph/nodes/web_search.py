from typing import Any, Dict

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_tavily import TavilySearch

from graph.state import GraphState

load_dotenv()

web_search_tool = TavilySearch(max_results=3)


def web_search(state: GraphState) -> Dict[str, Any]:
    """
    Performs a web search based on a user question if documents do not contain an answer.

    :param state: Question and documents
    :type state: GraphState
    :return: Documents and question
    :rtype: Dict[str, Any]
    """
    print("---WEBSEARCH---")
    question = state["question"]
    documents = state["documents"]

    tavily_search = web_search_tool.invoke({"query": question})

    tavily_results = tavily_search["results"]

    joined_tavily_result = "\n".join(
        [
            tavily_result["results"]
            for tavily_result in tavily_results
            if "results" in tavily_result
        ]
    )
    web_results = Document(page_content=joined_tavily_result)
    if documents is not None:
        documents.append(web_results)
    else:
        documents = [web_results]
    return {"documents": documents, "question": question}


if __name__ == "__main__":
    web_search(state={"question": "agent memory", "documents": None})

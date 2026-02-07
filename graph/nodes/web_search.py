from typing import Any, Dict

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_tavily import TavilySearch

from graph.state import GraphState

load_dotenv()

web_search_tool = TavilySearch(max_results=3)


def extract_web_content(results: list) -> str:
    """Extracts the content from a list of dictionaries and appends the text.

    Args:
        results (list): A list containing dictionaires.

    Returns:
        str: Appended text from the content key.
    """
    extracted_content = []
    for content in results:
        if isinstance(content, dict):
            for key, value in content.items():
                if key == "content":
                    extracted_content.append(value)
        else:
            print(f"Skipping non-dictionary item: {content}")
            return None
    joined_content = "\n".join(extracted_content)
    return joined_content


def web_search(state: GraphState) -> Dict[str, Any]:
    """Performs a web search based on a user question if documents do not contain an answer.

    Args:
        state (GraphState): The user question to query.

    Returns:
        Dict[str, Any]: Documents and question
    """
    print("---WEBSEARCH---")
    question = state["question"]
    documents = None

    tavily_search = web_search_tool.invoke({"query": question})
    tavily_results = tavily_search["results"]
    joined_tavily_result = extract_web_content(tavily_results)

    web_results = Document(page_content=joined_tavily_result)
    if documents is not None:
        documents.append(web_results)
    else:
        documents = [web_results]
    return {"documents": documents, "question": question}


if __name__ == "__main__":
    print(web_search(
        state={"question": "What is a pizza?", "documents": None}))

    print("Breakpoint")

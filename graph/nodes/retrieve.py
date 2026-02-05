from typing import Any, Dict

from graph.state import GraphState
from ingestion import retriever


def retrieve(state: GraphState) -> Dict[str, Any]:
    """
    Retrieves documents from vector database.

    :param state: Description
    :type state: GraphState
    :return: Description
    :rtype: Dict[str, Any]
    """
    print("---RETRIEVE---")
    question = state["question"]

    documents = retrieve.invoke(question)
    return {"documents": documents, "question": question}

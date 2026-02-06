from typing import Any, Dict

from graph.state import GraphState
from ingestion import retriever


def retrieve(state: GraphState) -> Dict[str, Any]:
    """Retrieves documents from vector database.

    Args:
        state (GraphState): User's question

    Returns:
        Dict[str, Any]: Relevant documents along with the question.
    """
    print("---RETRIEVE---")
    question = state["question"]

    documents = retriever.invoke(question)
    return {"documents": documents, "question": question}

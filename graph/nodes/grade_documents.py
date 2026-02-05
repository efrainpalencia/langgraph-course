from typing import Any, Dict

from graph.chains.retrieval_grader import retrieval_grader
from graph.state import GraphState


def grade_documents(state: GraphState) -> Dict[str, Any]:
    """
    Grades documents' relevance to a user's question with a score of yes or no.

    :param state: question and documents
    :type state: GraphState
    :return: Filtered documents, question, and web search flag
    :rtype: Dict[str, Any]
    """
    print("---CHECK DOCUMENT RELEVANCE TO QUESTION---")

    question = state["question"]
    documents = state["documents"]

    filtered_docs = []
    web_search = False

    for d in documents:
        score = retrieval_grader.invoke(
            {"question": question, "documents": d.page_content}
        )
        grade = score.binary_score
        if grade.lower() == "yes":
            print("---GRADE: DOCUMENT RELEVANT---")
            filtered_docs.append(d)
        else:
            print("---GRADE: DOCUMENT NOT RELEVANT")
            web_search = True
            continue
    return {"documents": filtered_docs, "question": question, "web_search": web_search}

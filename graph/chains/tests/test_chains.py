from pprint import pprint

from dotenv import load_dotenv

from graph.chains.generation import generation_chain
from graph.chains.retrieval_grader import GradeDocuments, retrieval_grader
from graph.chains.hallucination_grader import hallucination_grader, GradeHallucination
from graph.chains.answer_grader import answer_grader, GradeAnswer
from graph.chains.router import question_router, RouteQuery
from ingestion import retriever

load_dotenv()


def test_retrieval_grader_answer_yes() -> None:
    question = "short-term memory"
    docs = retriever.invoke(question)
    doc_text = docs[0].page_content

    res: GradeDocuments = retrieval_grader.invoke(
        {"question": question, "document": doc_text}
    )

    assert res.binary_score == "yes"


def test_retrieval_grader_answer_no() -> None:
    question = "how to make pizza"
    docs = retriever.invoke(question)
    doc_text = docs[0].page_content

    res: GradeDocuments = retrieval_grader.invoke(
        {"question": question, "document": doc_text}
    )

    assert res.binary_score == "no"


def test_generation() -> None:
    question = "short-term memory"
    docs = retriever.invoke(question)
    generation = generation_chain.invoke(
        {"context": docs, "question": question})
    pprint(generation)


def test_hallucination_grader_yes() -> None:
    question = "short-term memory"
    docs = retriever.invoke(question)
    generation = generation_chain.invoke(
        {
            "context": docs, "question": question
        }
    )
    res: GradeHallucination = hallucination_grader.invoke(
        {
            "documents": docs, "generation": generation
        }
    )
    assert res.binary_score


def test_hallucination_grader_no() -> None:
    question = "how to make pizza"
    docs = retriever.invoke(question)
    generation = generation_chain.invoke(
        {
            "context": docs, "question": question
        }
    )
    res: GradeHallucination = hallucination_grader.invoke(
        {
            "documents": docs, "generation": generation
        }
    )
    assert not res.binary_score


def test_answer_grader_yes() -> None:
    question = "short-term memory"
    docs = retriever.invoke(question)
    generation = generation_chain.invoke(
        {
            "context": docs, "question": question
        }
    )
    res: GradeAnswer = answer_grader.invoke(
        {"question": question, "generation": generation}
    )

    assert res.binary_score


def test_answer_grader_no() -> None:
    question = "how to make pizza"
    docs = retriever.invoke(question)
    generation = generation_chain.invoke(
        {
            "context": docs, "question": question
        }
    )
    res: GradeAnswer = answer_grader.invoke(
        {"question": question, "generation": generation}
    )

    assert not res.binary_score


def test_router_to_vectorstore() -> None:
    question = "short-term memory"

    res: RouteQuery = question_router.invoke(
        {"question": question}
    )

    assert res.datasource == "vectorstore"


def test_router_to_websearch() -> None:
    question = "how to make pizza"

    res: RouteQuery = question_router.invoke(
        {"question": question}
    )

    assert res.datasource == "websearch"

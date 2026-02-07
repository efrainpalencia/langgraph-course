from dotenv import load_dotenv
from langgraph.graph import END, StateGraph

from graph.consts import GENERATE, GRADE_DOCUMENTS, RETRIEVE, WEB_SEARCH
from graph.nodes import generate, grade_documents, retrieve, web_search
from graph.state import GraphState
from graph.chains.answer_grader import answer_grader
from graph.chains.hallucination_grader import hallucination_grader
from graph.chains.router import question_router, RouteQuery

load_dotenv()


def decide_to_generate(state):
    """Determines whether to generate a response or search the web for the condtional edge based on the web_search flag (yes or no).

    Args:
        state (str): Holds the state for the web_search

    Returns:
        (str): web_search or generate
    """
    print("---ASSESS GRADED DOCUMENTS---")

    if state["web_search"]:
        print(
            "---DECISION: NOT ALL DOCUMENTS ARE RELEVANT TO QUESTION, INCLUDED WEB SEARCH."
        )
        return WEB_SEARCH
    else:
        print("---DECISION: GENERATE---")
        return GENERATE


def grade_generation_grounded_in_documents_and_question(state: GraphState) -> str:
    # TODO: Docstring
    print("---CHECK HALLUCINATIONS---")
    question = state["question"]
    documents = state["documents"]
    generation = state["generation"]

    score = hallucination_grader.invoke(
        {"documents": documents, "generation": generation}
    )

    if hallucination_grade := score.binary_score:
        print("---DECISION: GENERATION IS GROUNDED IN DOCUMENTS---")
        print("---GRADE GENERATION VS QUESTION---")
        score = answer_grader.invoke(
            {"question": question, "generation": generation}
        )
        if answer_grade := score.binary_score:
            print("---DECISION: GENERATION ADDRESSES QUESTION---")
            return "useful"
        else:
            print("---DECISION: GENERATION DOES NOT ADDRESSES QUESTION---")
            return "not useful"
    else:
        print("---DECISION: GENERATION IS NOT GROUNDED IN DOCUMENTS, REGENERATE ANSWER")
        return "not supported"


def route_question(state: GraphState) -> str:
    # TODO: Docstring
    print("---ROUTE QUESTION---")

    question = state["question"]

    source: RouteQuery = question_router.invoke(
        {"question": question}
    )
    if source.datasource == "vectorstore":
        print("---ROUTING TO VECTORSTORE---")
        return RETRIEVE
    elif source.datasource == "websearch":
        print("---ROUTING TO WEB SEARCH---")
        return WEB_SEARCH


workflow = StateGraph(GraphState)

workflow.add_node(RETRIEVE, retrieve)
workflow.add_node(GENERATE, generate)
workflow.add_node(GRADE_DOCUMENTS, grade_documents)
workflow.add_node(WEB_SEARCH, web_search)
# workflow.set_entry_point(RETRIEVE)
workflow.add_edge(RETRIEVE, GRADE_DOCUMENTS)
workflow.set_conditional_entry_point(
    route_question,
    {
        WEB_SEARCH: WEB_SEARCH,
        RETRIEVE: RETRIEVE,
    },
)
workflow.add_conditional_edges(
    GRADE_DOCUMENTS,
    decide_to_generate,
    {
        WEB_SEARCH: WEB_SEARCH,
        GENERATE: GENERATE,
    },
)
workflow.add_conditional_edges(
    GENERATE,
    grade_generation_grounded_in_documents_and_question,
    {
        "useful": END,
        "not useful": WEB_SEARCH,
        "not supported": GENERATE,
    },
)
workflow.add_edge(WEB_SEARCH, GENERATE)
workflow.add_edge(GENERATE, END)

app = workflow.compile()

app.get_graph().draw_mermaid_png(output_file_path="graph.png")

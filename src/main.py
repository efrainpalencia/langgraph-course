from dotenv import load_dotenv

from typing import TypedDict, Annotated

from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.graph import END, StateGraph
from langgraph.graph import add_messages

from chains import generate_chain, reflect_chain

load_dotenv()


class MessageGraph(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


REFLECT = "reflect"
GENERATE = "generate"


def generatation_node(state: MessageGraph):
    return {"messages": [generate_chain.invoke({"messages": state["messages"]})]}


def reflection_node(state: MessageGraph):
    res = reflect_chain.invoke({"messages": state["messages"]})
    return {"messages": [HumanMessage(content=res.content)]}


builder = StateGraph(state_schema=MessageGraph)
builder.add_node(GENERATE, generatation_node)
builder.add_node(REFLECT, reflection_node)
builder.set_entry_point(GENERATE)


def should_continue(state: MessageGraph):
    if len(state["messages"]) > 6:
        return END
    return REFLECT


builder.add_conditional_edges(GENERATE, should_continue, path_map={
    END: END,
    REFLECT: REFLECT
})
builder.add_edge(REFLECT, GENERATE)

graph = builder.compile()
print(graph.get_graph().draw_mermaid_png(output_file_path="flow.png"))

if __name__ == '__main__':
    print("Reasoning Agent")
    inputs = HumanMessage(content=""" @TechCrunch
                                Elon Musk’s SpaceX officially acquires Elon Musk’s xAI, with plan to build data centers in space
                                """)

    response = graph.invoke(inputs)

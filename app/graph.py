from typing import TypedDict, Literal
from langgraph.graph import StateGraph, START, END
from app.nodes import (
    router, retrieve_context_q_n_a, retrieve_context_medical_device, 
    retrieve_context_drugs, tavily_web_search, 
    check_context_relevance, build_prompt, call_llm
)

class GraphState(TypedDict):
    query: str
    context: str
    prompt: str
    response: str
    source: str
    is_relevant: str
    iteration_count: int
    url: str

def route_decision(state: GraphState) -> str:
    return state["source"]

def relevance_decision(state: GraphState) -> str:
    return state["is_relevant"]

workflow = StateGraph(GraphState)

# Add Nodes
workflow.add_node("Router", router)
workflow.add_node("Retrieve_QnA", retrieve_context_q_n_a)
workflow.add_node("Retrieve_Device", retrieve_context_medical_device)
workflow.add_node("Retrieve_Drugs", retrieve_context_drugs)  # NEW NODE
workflow.add_node("Web_Search", tavily_web_search)
workflow.add_node("Relevance_Checker", check_context_relevance)
workflow.add_node("Augment", build_prompt)
workflow.add_node("Generate", call_llm)

# Add Edges
workflow.add_edge(START, "Router")

workflow.add_conditional_edges(
    "Router",
    route_decision,
    {
        "Retrieve_QnA": "Retrieve_QnA",
        "Retrieve_Device": "Retrieve_Device",
        "Retrieve_Drugs": "Retrieve_Drugs",
        "Web_Search": "Web_Search",
    },
)

# Wire all retrievers to relevance checker
for node in ["Retrieve_QnA", "Retrieve_Device", "Retrieve_Drugs", "Web_Search"]:
    workflow.add_edge(node, "Relevance_Checker")

workflow.add_conditional_edges(
    "Relevance_Checker",
    relevance_decision,
    {
        "Yes": "Augment",
        "No" : "Web_Search",
    }
)

workflow.add_edge("Augment", "Generate")
workflow.add_edge("Generate", END)

# Compile Graph
agentic_rag = workflow.compile()

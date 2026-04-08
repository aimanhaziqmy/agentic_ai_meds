import os
from typing import Literal
from langchain_community.tools.tavily_search import TavilySearchResults
from app.llm import get_llm_response
from app.database import collection_qna, collection_device, collection_drugs

def retrieve_context_q_n_a(state):
    
    print("---RETRIEVING CONTEXT: Q&A---")
    results = collection_qna.query(query_texts=[state["query"]], n_results=3)
    state["context"] = "\n".join(results["documents"][0]) if results["documents"] else "No documents found."
    state["source"] = "Medical Q&A Collection"
    return state

def retrieve_context_medical_device(state):
    print("---RETRIEVING CONTEXT: Device Manual---")
    results = collection_device.query(query_texts=[state["query"]], n_results=3)
    state["context"] = "\n".join(results["documents"][0]) if results["documents"] else "No documents found."
    state["source"] = "Medical Device Manual"
    return state

# Retrieve Drugs
def retrieve_context_drugs(state):
    print("---RETRIEVING CONTEXT: Drugs---")
    results = collection_drugs.query(query_texts=[state["query"]], n_results=3)
    state["context"] = "\n".join(results["documents"][0]) if results["documents"] else "No documents found."
    state["source"] = "Drugs Dataset"
    return state

def tavily_web_search(state):
    print("---RETRIEVING CONTEXT: Web Search---")
    tavily_search = TavilySearchResults(max_results=2)
    result_ = tavily_search.invoke({"query": state['query']})
    
    if result_:
        state["context"] = result_[0]['content']
        state["url"] = result_[0].get('url', 'External Web Source')
    else:
        state["context"] = "No relevant web results found."
        state["url"] = "None"
        
    state["source"] = "Web Search"
    return state

def router(state) -> dict:
    query = state["query"]
    decision_prompt = f"""
    You are a routing agent. Based on the user query, decide where to look for information.
    Options:
    - Retrieve_QnA : general medical knowledge, symptoms, diseases, or treatment.
    - Retrieve_Device : medical devices, manuals, machine models, or instructions.
    - Retrieve_Drugs : medicine names, drug composition, side effects, uses, or reviews.
    - Web_Search : recent news, brand names, or external data not covered above.

    Query : "{query}"
    Respond ONLY with one of: Retrieve_QnA, Retrieve_Device, Retrieve_Drugs, Web_Search 
    """
    router_decision = get_llm_response(decision_prompt).strip()
    print(f"---ROUTER DECISION: {router_decision}---")
    
    # Fallback to Web_Search if LLM hallucinates format
    valid_decisions = ["Retrieve_QnA", "Retrieve_Device", "Retrieve_Drugs", "Web_Search"]
    if router_decision not in valid_decisions:
        router_decision = "Web_Search"

    state["source"] = router_decision
    return state

def check_context_relevance(state):
    print("---CHECKING RELEVANCE---")
    iteration_count = state.get("iteration_count", 0) + 1
    state["iteration_count"] = iteration_count

    if iteration_count >= 3:
        print("--- MAX ITERATIONS REACHED, FORCING 'Yes' ---")
        state["is_relevant"] = "Yes"
        return state

    relevance_prompt = f"""
    Check if the context below is relevant to the user query.
    Context: {state['context']}
    User Query: {state['query']}
    Options:
    - Yes: context is relevant to the question.
    - No: context is completely irrelevant.
    Answer with only 'Yes' or 'No'.
    """
    decision = get_llm_response(relevance_prompt).strip()
    print(f"---RELEVANCE DECISION: {decision}---")
    state["is_relevant"] = decision
    return state

def build_prompt(state):
    print("---AUGMENTING PROMPT---")
    source = state.get("source", "Unknown Source")
    url = state.get("url", "")

    if source == "Web Search":
        citation_rules = f"Include a brief in-text citation (e.g., [1]) and append '\n\nSource: {url}' at the end."
    else:
        citation_rules = f"Do not use in-text citations. Append '\n\nSource: {source}' at the end."

    state["prompt"] = f"""
    Answer the following question using the context below.
    Context: {state['context']}
    Question: {state['query']}
    Instructions:
    - Limit your answer to 50 words.
    - {citation_rules}
    """
    return state

def call_llm(state):
    print("---GENERATING RESPONSE---")
    state["response"] = get_llm_response(state["prompt"])
    return state
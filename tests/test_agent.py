import pytest
from app.nodes import router

def test_router_selects_drugs():
    state = {"query": "What are the side effects of ibuprofen?"}
    updated_state = router(state)
    assert updated_state["source"] == "Retrieve_Drugs", f"Expected 'Retrieve_Drugs' but got '{updated_state['source']}'"

def test_router_selects_device():
    state = {"query": "How do I operate the Medtronic insulin pump?"}
    updated_state = router(state)
    assert updated_state["source"] == "Retrieve_Device", f"Expected 'Retrieve_Device' but got '{updated_state['source']}'"

def test_router_selects_qna():
    state = {"query": "What are the symptoms of diabetes?"}
    updated_state = router(state)
    assert updated_state["source"] == "Retrieve_QnA", f"Expected 'Retrieve_QnA' but got '{updated_state['source']}'"

def test_router_selects_web_search():
    state = {"query": "What is the latest research on Alzheimer's treatment?"}
    updated_state = router(state)
    assert updated_state["source"] == "Web_Search", f"Expected 'Web_Search' but got '{updated_state['source']}'"



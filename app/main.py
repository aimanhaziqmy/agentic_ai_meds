import streamlit as st
from app.graph import agentic_rag

st.set_page_config(page_title="Medical Agentic RAG", page_icon="⚕️")
st.title("⚕️ Medical AI Assistant")
st.caption("Powered by LangGraph, ChromaDB, Tavily Web Search, and 3 Specialized Datasets")

if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hello! I can answer medical questions, check device manuals, or look up drug information. How can I help?"}
    ]

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Ask a medical question..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.status("Agent processing request...", expanded=True) as status:
            
            input_state = {
                "query": prompt, 
                "iteration_count": 0,
                "context": "", "prompt": "", "response": "", 
                "source": "", "url": "", "is_relevant": ""
            }
            
            final_response = ""
            
            # Stream the LangGraph execution
            for step in agentic_rag.stream(input_state):
                for key, value in step.items():
                    st.write(f"✅ Finished step: **{key}**")
                    
                    if key == "Router":
                        st.write(f"🧭 Routed to: **{value.get('source')}**")
                    if key == "Relevance_Checker":
                        st.write(f"🧐 Relevance Check: **{value.get('is_relevant', 'Unknown')}**")
                    
                    if "response" in value and value["response"]:
                        final_response = value["response"]
            
            status.update(label="Answer generated!", state="complete", expanded=False)
        
        st.markdown(final_response)
        
    st.session_state.messages.append({"role": "assistant", "content": final_response})
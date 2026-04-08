# Agentic AI Meds

A prototype Agentic RAG application for querying medical data. Built with LangChain, Streamlit, Docker, and Traefik.

## Quick Start

1. **Clone the repo**

```bash
   git clone git@github.com:aimanhaziqmy/agentic_ai_meds.git
   cd agentic_ai_meds
   ```

2. **Set your API Key**
   Create a `.env` file in the root directory and add your OpenAI key:
   ```env
   OPENAI_API_KEY=your_api_key_here
   TAVILY_API_KEY=your_api_key_here

   ```

3. **Run with Docker**
   ```bash
   docker-compose up -d --build
   ```

4. **Open the App**
   Navigate to `http://localhost:8501` (if you use port number) in your browser to interact with the agent.

## Tech Stack
* **LLM Orchestration:** LangChain
* **Vector DB:** ChromaDB
* **UI:** Streamlit
* **Deployment:** Docker & Traefik



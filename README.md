title: RAG Powerd Chat Bot
emoji: 🔥
colorFrom: gray
colorTo: blue
sdk: gradio
sdk_version: 5.49.1
app_file: app.py
pinned: false
license: apache-2.0
short_description: RAG Powerd Chat Bot

# RAG-Powered Chatbot (FAISS + MiniLM + Gradio)

## Quickstart (local)
```bash
python -m venv .venv && source .venv/bin/activate    # (Windows: .venv\Scripts\activate)
pip install -r requirements.txt
mkdir -p docs storage
# add some PDFs/TXT into docs/
python ingest.py
python app.py

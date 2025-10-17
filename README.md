# RAG-Powered Chatbot (FAISS + MiniLM + Gradio)

## Quickstart (local)
```bash
python -m venv .venv && source .venv/bin/activate    # (Windows: .venv\Scripts\activate)
pip install -r requirements.txt
mkdir -p docs storage
# add some PDFs/TXT into docs/
python ingest.py
python app.py
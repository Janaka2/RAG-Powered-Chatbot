---
title: RAG_Powerd_Chat_Bot
emoji: 🤖
colorFrom: indigo
colorTo: purple
sdk: gradio
# Use the version you install in requirements.txt (or omit if unsure)
sdk_version: "4.44.0"
app_file: app.py
pinned: false
---
# RAG-Powered Chatbot (FAISS + MiniLM + Gradio)

## Quickstart (local)
```bash
python -m venv .venv && source .venv/bin/activate    # (Windows: .venv\Scripts\activate)
pip install -r requirements.txt
mkdir -p docs storage
# add some PDFs/TXT into docs/
python ingest.py
python app.py

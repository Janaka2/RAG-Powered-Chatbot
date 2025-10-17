import os
import gradio as gr
from rag_core import RAGPipeline

# --- Config & guards
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
if not OPENAI_API_KEY:
    raise RuntimeError("Please set OPENAI_API_KEY in your environment/Space secrets.")

rag = RAGPipeline(
    index_dir="storage",
    docs_dir="docs",
    embed_model_name="sentence-transformers/all-MiniLM-L6-v2",
    openai_model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
    top_k=int(os.getenv("TOP_K", "5")),
    mmr_lambda=float(os.getenv("MMR_LAMBDA", "0.5")),
)

SYSTEM_HINT = (
    "You are a precise RAG assistant. Use ONLY the provided context. "
    "If the answer isn't in the context, say \"I don't know.\" Cite sources like [1], [2]."
)

def _to_paths(files):
    """Normalize Gradio Files input (varies by version) into a list of existing file paths."""
    paths = []
    for f in files or []:
        # Newer Gradio may hand us str path; older gives file-like with .name
        if isinstance(f, (str, os.PathLike)):
            p = str(f)
        else:
            p = getattr(f, "name", None)
        if p and os.path.exists(p):
            paths.append(p)
    return paths

def chat(user_msg, history, files):
    upload_report = None

    # Normalize uploads and index them
    paths = _to_paths(files)
    if paths:
        added = rag.add_files(paths)  # expects list[str]
        rag.save()
        if added:
            upload_report = "📚 Indexed: " + ", ".join(os.path.basename(p) for p in added)

    # Keep last few exchanges only (short context to the LLM)
    trimmed_history = history[-4:] if history else []
    answer, citations = rag.answer(
        user_msg,
        chat_history=trimmed_history,
        system_hint=SYSTEM_HINT
    )

    display = answer
    if citations:
        display += "\n\nSources: " + " ".join(
            f"[{i+1}] {c.get('title','')}" for i, c in enumerate(citations)
        )
    if upload_report:
        display = upload_report + "\n\n" + display

    # IMPORTANT: return just a string for ChatInterface to avoid schema bugs
    return display

with gr.Blocks(title="RAG-Powered Chatbot • OpenAI") as demo:
    gr.Markdown("## 🔵 RAG-Powered Chatbot (OpenAI)")
    with gr.Row():
        with gr.Column(scale=3):
            gr.ChatInterface(
                fn=chat,
                textbox=gr.Textbox(
                    placeholder="Ask about your documents…",
                    lines=2
                ),
                # Use Files (not File). We avoid 'types="filepath"' for max compatibility.
                additional_inputs=[
                    gr.Files(
                        label="Upload files (optional)",
                        file_count="multiple",
                        file_types=[".pdf", ".txt", ".md"],
                    )
                ],
            )
        with gr.Column(scale=1):
            gr.Markdown("### Admin")
            btn_reindex = gr.Button("🧰 Rebuild index from /docs")
            out = gr.Markdown()

    def rebuild():
        n = rag.rebuild_from_folder()
        rag.save()
        return f"Rebuilt index from /docs. Chunks: {n}"

    btn_reindex.click(rebuild, outputs=out)

if __name__ == "__main__":
    # Bind to loopback to avoid localhost/proxy issues on Windows/VPNs
    demo.queue().launch(server_name="127.0.0.1", server_port=7860, share=False)
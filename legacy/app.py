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
    """Normalize Gradio Files input into a list of existing file paths."""
    paths = []
    for f in files or []:
        if isinstance(f, (str, os.PathLike)):
            p = str(f)
        else:
            p = getattr(f, "name", None)
        if p and os.path.exists(p):
            paths.append(p)
    return paths

def _format_answer(answer, citations, upload_report=None):
    display = answer
    if citations:
        display += "\n\nSources: " + " ".join(
            f"[{i+1}] {c.get('title','')}" for i, c in enumerate(citations)
        )
    if upload_report:
        display = upload_report + "\n\n" + display
    return display

# ---- Event handlers using Chatbot type="messages" ----
def on_user_submit(user_msg, messages):
    """
    messages: list[dict] like [{"role":"user","content":"..."}, {"role":"assistant","content":"..."}]
    """
    messages = messages or []
    messages.append({"role": "user", "content": user_msg})
    return "", messages

def on_bot_respond(messages, files):
    upload_report = None
    paths = _to_paths(files)
    if paths:
        added = rag.add_files(paths)
        rag.save()
        if added:
            upload_report = "📚 Indexed: " + ", ".join(os.path.basename(p) for p in added)

    # Build trimmed chat history as tuples for your pipeline (user, assistant)
    # Convert messages list to tuples [(u1,a1), (u2,a2), ...] but we only keep last 4 pairs
    pairs = []
    last_user = None
    for m in messages[:-1]:  # exclude the last user just submitted
        if m["role"] == "user":
            last_user = m["content"]
        elif m["role"] == "assistant" and last_user is not None:
            pairs.append((last_user, m["content"]))
            last_user = None
    trimmed = pairs[-4:] if pairs else []

    user_msg = messages[-1]["content"]  # the latest user turn

    answer, citations = rag.answer(
        user_msg, chat_history=trimmed, system_hint=SYSTEM_HINT
    )

    messages.append({"role": "assistant", "content": _format_answer(answer, citations, upload_report)})
    return messages

# ---------- UI ----------
with gr.Blocks(title="RAG-Powered Chatbot • OpenAI", fill_height=True) as demo:
    gr.Markdown("## 🔵 RAG-Powered Chatbot (OpenAI)")
    with gr.Row():
        with gr.Column(scale=3):
            chatbot = gr.Chatbot(height=480, label="Chatbot", type="messages")

            files = gr.Files(
                label="Upload files (optional)",
                type="filepath",
                file_count="multiple",
                file_types=[".pdf", ".txt", ".md"],
            )

            with gr.Row():
                textbox = gr.Textbox(
                    placeholder="Ask about your documents…",
                    lines=2,
                    autofocus=True,
                    scale=10,
                )
                send_btn = gr.Button("Send", scale=1)

            # Enter → send
            submit_evt = textbox.submit(
                on_user_submit, inputs=[textbox, chatbot], outputs=[textbox, chatbot]
            )
            # Click Send → send
            send_evt = send_btn.click(
                on_user_submit, inputs=[textbox, chatbot], outputs=[textbox, chatbot]
            )
            # Then bot responds for both
            for evt in (submit_evt, send_evt):
                evt.then(on_bot_respond, inputs=[chatbot, files], outputs=[chatbot])

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
    demo.queue().launch(server_name="127.0.0.1", server_port=7860, share=False)
import os
import gradio as gr
from config import Settings
from core.pipeline import RAGPipeline

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
if not OPENAI_API_KEY:
    raise RuntimeError("Please set OPENAI_API_KEY in your environment.")

settings = Settings()
rag = RAGPipeline(settings=settings)

SYSTEM_HINT = (
    'You are a precise RAG assistant. Use ONLY the provided context. '
    'If the answer isn\'t in the context, say "I don\'t know." Cite sources like [1], [2].'
)

def _to_paths(files):
    paths = []
    for f in files or []:
        p = getattr(f, "name", None) if not isinstance(f, (str, os.PathLike)) else str(f)
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

def on_user_submit(user_msg, messages):
    messages = messages or []
    messages.append({"role": "user", "content": user_msg})
    return "", messages

def on_bot_respond(messages, files, retriever, k, mmr_lambda, hybrid_alpha):
    upload_report = None
    # override runtime settings
    rag.cfg.retrieval.top_k = int(k)
    rag.cfg.retrieval.mmr_lambda = float(mmr_lambda)
    rag.cfg.retrieval.hybrid_alpha = float(hybrid_alpha)

    paths = _to_paths(files)
    if paths:
        added = rag.add_files(paths)
        if added:
            upload_report = "📚 Indexed: " + ", ".join(os.path.basename(p) for p in added)

    pairs = []
    last_user = None
    for m in messages[:-1]:
        if m["role"] == "user":
            last_user = m["content"]
        elif m["role"] == "assistant" and last_user is not None:
            pairs.append((last_user, m["content"]))
            last_user = None
    trimmed = pairs[-4:] if pairs else []

    user_msg = messages[-1]["content"]
    answer, citations = rag.answer(
        user_msg,
        retriever=retriever,
        system_hint=SYSTEM_HINT,
        chat_history=trimmed,
    )
    messages.append({"role": "assistant", "content": _format_answer(answer, citations, upload_report)})
    return messages

with gr.Blocks(title="RAG • Modular Skeleton", fill_height=True) as demo:
    gr.Markdown("## 🔵 RAG • Modular Skeleton (FAISS + BM25 + MMR/Hybrid)")
    with gr.Row():
        with gr.Column(scale=3):
            chatbot = gr.Chatbot(height=500, label="Chatbot", type="messages")

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

            with gr.Accordion("⚙️ Retrieval options", open=False):
                retriever = gr.Radio(
                    choices=["mmr", "similarity", "hybrid"],
                    value="mmr",
                    label="Retriever",
                )
                k = gr.Slider(1, 10, value=5, step=1, label="Top-K")
                mmr_lambda = gr.Slider(0.0, 1.0, value=0.5, step=0.05, label="MMR λ (diversity vs relevance)")
                hybrid_alpha = gr.Slider(0.0, 1.0, value=0.6, step=0.05, label="Hybrid α (dense weight)")

            submit_evt = textbox.submit(on_user_submit, inputs=[textbox, chatbot], outputs=[textbox, chatbot])
            send_evt = send_btn.click(on_user_submit, inputs=[textbox, chatbot], outputs=[textbox, chatbot])
            for evt in (submit_evt, send_evt):
                evt.then(on_bot_respond, inputs=[chatbot, files, retriever, k, mmr_lambda, hybrid_alpha], outputs=[chatbot])

        with gr.Column(scale=1):
            gr.Markdown("### Admin")
            btn_reindex = gr.Button("🧰 Rebuild index from /docs")
            out = gr.Markdown()

            def rebuild():
                n = rag.rebuild_from_folder()
                return f"Rebuilt index from /docs. Chunks: {n}"

            btn_reindex.click(rebuild, outputs=out)

if __name__ == "__main__":
    demo.queue().launch(
        server_name="0.0.0.0",   # avoid proxy/localhost checks
        server_port=7860,
        share=True,              # or False if localhost works for you
        show_api=False,          # <-- key workaround for the schema bug
        inbrowser=False,
    )
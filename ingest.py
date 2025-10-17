from rag_core import RAGPipeline

if __name__ == "__main__":
    rag = RAGPipeline(index_dir="storage",
                      docs_dir="docs",
                      embed_model_name="sentence-transformers/all-MiniLM-L6-v2")
    n = rag.rebuild_from_folder()
    rag.save()
    print(f"Built index from /docs. Total chunks: {n}")

from config import Settings
from core.pipeline import RAGPipeline

if __name__ == "__main__":
    rag = RAGPipeline(Settings())
    n = rag.rebuild_from_folder()
    print(f"Built index from /docs. Total chunks: {n}")

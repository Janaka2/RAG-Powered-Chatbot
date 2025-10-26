
"""
Compatibility layer so existing imports like `from rag_core import RAGPipeline`
continue to work. Under the hood it delegates to core.pipeline.RAGPipeline.
"""
from dataclasses import dataclass
from typing import Optional, List, Tuple, Dict

# Re-export
from core.pipeline import RAGPipeline  # noqa: F401


# Migration

This fork preserves your previous API surface while upgrading to a modular layout.

## Keep your old imports working
```python
from rag_core import RAGPipeline   # now delegates to core.pipeline.RAGPipeline
import utils                      # functions redirect to core/* modules
```

## Recommended next steps
- Gradually replace `utils.*` calls with direct imports from `core.utils`, `core.chunking`, etc.
- If you had custom logic in your original `rag_core.py`, port it into `core/pipeline.py` in small steps.
- Add Cross-Encoder re-ranking and RRF fusion options (scaffold included).

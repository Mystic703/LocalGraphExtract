# KG Pipeline — Local

Document → Knowledge Graph (Neo4j) → Hybrid RAG (fully local, no API keys)

## Project structure

```
GraphRag Local/
├── config.py       # All settings, switch MODE here
├── loader.py       # PDF → Markdown → LlamaIndex nodes
├── models.py       # LLM + embedding singletons
├── prompts.py      # All prompt templates
├── graph.py        # Neo4j store, index building, triplet validation
├── retrieval.py    # Hybrid retriever + query engine
├── visualize.py    # NetworkX graph plot
├── main.py         # CLI entry point
├── data/           # Place your PDF files here
├── models/         # Place your GGUF model files here
├── cache/          # Auto-generated doc cache
└── logs/           # Pipeline + triplet logs
```

## Quickstart

```bash
# 1. Install dependencies
pip install docling llama-index llama-index-llms-llama-cpp \
    llama-index-embeddings-huggingface llama-index-graph-stores-neo4j \
    networkx matplotlib nest_asyncio

# 2. Start Neo4j (Docker)
docker run -d --name neo4j \
  -e NEO4J_AUTH=neo4j/nonenone \
  -p 7474:7474 -p 7687:7687 \
  neo4j:latest

# 3. Download GGUF models and place them in models/
#    - models/Llama-3.2-3B-Instruct-Q4_K_M.gguf  (extraction)
#    - models/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf  (answers)

# 4. Edit config.py if needed
#    - Adjust Neo4j credentials
#    - Change model paths

# 5. Run
python main.py --source path/to/document.pdf          # full run
python main.py --source path/to/document.pdf --viz    # + visualise graph
python main.py --query-only                            # skip ingestion
python main.py --source path/to/document.pdf --ingest # ingest only
```

## Switching to production (larger models)

In `config.py`, change one line:

```python
MODE = "local_server"   # was "local_test"
```

Everything else (model paths, thread counts, context window, batch size)
updates automatically from the `_PROFILES` dict.

## Anti-hallucination measures

1. **No few-shot value examples** in the extraction prompt — prevents the model
   anchoring on example numbers.
2. **Verbatim-only instruction** — the model is explicitly told not to generate
   values absent from the source text.
3. **Post-extraction validation** (`graph.py:_validate_node_triplets`) — any
   numeric object that cannot be found verbatim in the source chunk is removed
   before the triplet is committed to Neo4j.
4. **Triplet logging** (`logs/triplet_log.jsonl`) — every extraction is logged
   with its source chunk so you can audit and diff quality between model runs.

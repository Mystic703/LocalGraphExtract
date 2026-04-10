# KG Pipeline

Document → Knowledge Graph (Neo4j) → Hybrid RAG

## Project structure

```
kg_pipeline/
├── config.py               # All settings, switch MODE here
├── loader.py               # PDF → Markdown → LlamaIndex nodes
├── models.py               # LLM + embedding singletons
├── prompts.py              # All prompt templates
├── graph.py                # Neo4j store and index building
├── retrieval.py            # Hybrid retriever + query engine
├── visualize.py            # NetworkX graph plot
├── main.py                 # CLI entry point
└── OFFLINE_DEPLOYMENT.md  # Guide for air-gapped deployment (CCRT/TGCC)
```

---

## Quickstart

```bash
# 1. Install dependencies
pip install -r requirements.txt
CMAKE_ARGS="-DLLAMA_BLAS=ON -DLLAMA_BLAS_VENDOR=OpenBLAS" pip install llama-cpp-python

# 2. Start Neo4j
docker run -d --name neo4j \
  -e NEO4J_AUTH=neo4j/nonenone \
  -p 7474:7474 -p 7687:7687 \
  neo4j:5.18

# 3. Place your GGUF models in the models/ folder

# 4. Run
python main.py --source path/to/document.pdf
```

---

## CLI usage

```bash
# Ingest a document then run demo questions
python main.py --source path/to/document.pdf

# Ingest then start interactive Q&A
python main.py --source path/to/document.pdf --interactive

# Skip ingestion, load existing index, interactive Q&A
python main.py --interactive

# Skip ingestion, load existing index, run demo questions
python main.py --query-only

# Ingest only, no Q&A
python main.py --source path/to/document.pdf --ingest

# Ingest and visualise the graph
python main.py --source path/to/document.pdf --viz

# Force re-convert PDF (ignore cache)
python main.py --source path/to/document.pdf --force-reload
```

---

## Switching environments

In `config.py`, change one line:

```python
MODE = "local_test"    # development — 3B model, 20 nodes
MODE = "local_server"  # production  — large models, all nodes
```

Everything else (model paths, thread counts, context window, batch size)
updates automatically from the `_PROFILES` dict.

---

## Air-gapped deployment (CCRT/TGCC)

See `OFFLINE_DEPLOYMENT.md`.
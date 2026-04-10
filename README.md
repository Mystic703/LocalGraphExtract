# GraphRAG — Local & CCRT

Document → Knowledge Graph → Hybrid RAG  
Runs offline on a laptop (llama.cpp) **or** on CEA CCRT A100 GPUs (vLLM).  
No API keys. No internet required at inference time.

---

## Project structure

```
GraphRag Local/
├── config.py               # All settings — set MODE here (or via env var)
├── models.py               # LLM + embedding factories (llama.cpp / vLLM / HF)
├── graph.py                # Graph store (Neo4j or file-based SimplePropertyGraphStore)
├── loader.py               # PDF → Markdown (Docling + pypdf fallback), chunking
├── retrieval.py            # Hybrid vector+graph query engine
├── prompts.py              # All prompt templates
├── visualize.py            # NetworkX graph plot (local only)
├── main.py                 # CLI entry point
│
├── scripts/
│   ├── ingest.slurm        # SLURM job — ingestion on CCRT
│   ├── query.slurm         # SLURM job — query on CCRT
│   ├── setup_env.sh        # One-time virtualenv setup on CCRT
│   └── download_models.py  # Pre-download HF models before going offline
│
├── requirements-local.txt  # Laptop / workstation
├── requirements-hpc.txt    # CCRT A100 (vLLM)
│
├── data/                   # Source PDFs
├── models/                 # GGUF files (local) or HF snapshot cache (CCRT)
├── cache/                  # Doc cache (Markdown) + graph_store.json
└── logs/                   # Pipeline logs (one file per SLURM job)
```

---

## Mode 1 — Local laptop (`local_test`)

```bash
# 1. Install
pip install -r requirements-local.txt

# 2. Start Neo4j
docker run -d --name neo4j \
  -e NEO4J_AUTH=neo4j/nonenone \
  -p 7474:7474 -p 7687:7687 \
  neo4j:latest

# 3. Download GGUF models → models/
#    huggingface-cli download bartowski/Llama-3.2-3B-Instruct-GGUF \
#        Llama-3.2-3B-Instruct-Q4_K_M.gguf --local-dir models/
#    huggingface-cli download bartowski/Meta-Llama-3.1-8B-Instruct-GGUF \
#        Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf --local-dir models/

# 4. Run (MODE defaults to "local_test" in config.py)
python main.py --source data/rapport.pdf --ingest   # ingest only
python main.py --query-only                          # load graph + query
python main.py --source data/rapport.pdf             # ingest + query
python main.py --source data/rapport.pdf --viz       # + graph plot
```

---

## Mode 2 — CEA CCRT A100 (`ccrt`)

### Architecture differences vs local

| | `local_test` | `ccrt` |
|---|---|---|
| LLM backend | llama.cpp (CPU/GGUF) | **vLLM** (A100, bfloat16) |
| Graph store | Neo4j (Docker) | **SimplePropertyGraphStore** (JSON file, no server) |
| Extraction model | Llama-3.2-3B Q4 GGUF | `meta-llama/Llama-3.2-3B-Instruct` |
| Answer model | Llama-3.1-8B Q4 GGUF | `meta-llama/Llama-3.1-70B-Instruct` (4 GPUs) |
| Internet at runtime | optional | **blocked** — fully offline |

### First-time setup (login node, with internet)

```bash
# 1. Copy project to CCRT
rsync -av "GraphRag Local/" ccrt:$WORK/GraphRag_Local/

# 2. Create virtualenv + install deps
ssh ccrt "cd $WORK/GraphRag_Local && bash scripts/setup_env.sh"

# 3. Download models (still on login node = internet access)
ssh ccrt "cd $WORK/GraphRag_Local && source .venv/bin/activate && \
    python scripts/download_models.py --mode ccrt --models-dir $WORK/models"

# 4. Copy your PDF
scp rapport.pdf ccrt:$WORK/data/
```

### Running jobs

```bash
# Ingest (~ 2-4 h depending on PDF size)
sbatch scripts/ingest.slurm

# Check progress
tail -f logs/ingest_<JOBID>.out

# Query (once ingest is done — graph_store.json is in $SCRATCH/graphrag_cache/)
sbatch scripts/query.slurm "Comment est assurée la protection contre les incendies ?"
```

### GPU allocation

| Model | Size | GPUs needed |
|---|---|---|
| Llama-3.2-3B (extraction) | ~6 GB | 1 × A100-80G |
| Llama-3.1-70B (answers) | ~35 GB bfloat16 | 4 × A100-80G |

Both models are **not loaded simultaneously** — extraction runs first (ingest job),  
then only the answer model is loaded (query job). `#SBATCH --gres=gpu:4` covers both jobs.

---

## Switching modes

In `config.py`:
```python
MODE = "ccrt"       # "local_test" | "ccrt"
```

Or via environment variable (used by SLURM scripts automatically):
```bash
PIPELINE_MODE=ccrt python main.py --query-only
```

---

## Anti-hallucination measures

1. **No few-shot value examples** — extraction prompt contains no concrete numbers,
   preventing the model from anchoring on example values.
2. **Verbatim-only instruction** — the model is explicitly told every term must appear
   in the source text as-is.
3. **Checkpoint after every node** — `graph_store.json` is flushed to disk after each
   node so a SLURM timeout doesn't lose progress.
4. **pypdf fallback** — pages Docling can't parse are recovered via pypdf rather than
   silently dropped.

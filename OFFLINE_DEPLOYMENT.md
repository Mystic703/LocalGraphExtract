# Déploiement Offline — CCRT/TGCC

Guide de préparation et déploiement du pipeline KG sur environnement air-gappé
(serveurs sans accès internet).

---

## Contexte

Le pipeline utilise plusieurs composants qui téléchargent des ressources au premier
lancement. En environnement air-gappé, tout doit être préparé sur une machine
connectée et transféré manuellement.

Ressources à préparer :
- Modèles LLM (GGUF)
- Modèle d'embedding (BAAI/bge-m3)
- Modèles Docling (layout + OCR)
- Dépendances Python (pip packages)

---

## Étape 1 — Sur une machine connectée à internet

### 1.1 Installer les outils de téléchargement

```bash
pip install huggingface_hub
```

### 1.2 Télécharger les modèles LLM (GGUF)

Choisir la taille selon la RAM disponible sur le serveur cible :

| Modèle | Taille Q4_K_M | RAM recommandée |
|---|---|---|
| Qwen2.5-7B | ~5 GB | 16 GB |
| Qwen2.5-14B | ~9 GB | 32 GB |
| Qwen2.5-32B | ~20 GB | 48 GB |
| Qwen2.5-72B | ~43 GB | 80 GB+ |

```bash
# Exemple avec 72B (adapter selon les ressources disponibles)
huggingface-cli download bartowski/Qwen2.5-72B-Instruct-GGUF --include "Qwen2.5-72B-Instruct-Q4_K_M.gguf" --local-dir models/
```

### 1.3 Télécharger le modèle d'embedding

```bash
huggingface-cli download BAAI/bge-m3 --local-dir models/embeddings/bge-m3
```

### 1.4 Télécharger les modèles Docling

```bash
huggingface-cli download ds4sd/docling-models --local-dir models/docling/docling-models

huggingface-cli download docling-project/docling-layout-heron --local-dir models/docling/docling-layout-heron
```

### 1.5 Télécharger les dépendances Python

```bash
pip download -r requirements.txt -d ./pip_packages/

# Wheel llama-cpp-python précompilée pour Linux CPU
pip download llama-cpp-python \
  --only-binary=:all: \
  --platform linux_x86_64 \
  --python-version 311 \
  -d ./pip_packages/
```
### 1.6 Sauvegarder l'image Docker Neo4j

```bash
docker pull neo4j:5.18
docker save neo4j:5.18 -o neo4j-5.18.tar
```

### 1.7 Structure du bundle à transférer

```
transfer_bundle/
├── models/
│   ├── Qwen2.5-72B-Instruct-Q4_K_M.gguf
│   ├── embeddings/
│   │   └── bge-m3/
│   └── docling/
│       ├── docling-models/
│       └── docling-layout-heron/
├── pip_packages/
├── neo4j-5.18.tar
├── kg_pipeline/              ← code source, copier tel quel
└── data/                     ← PDFs à traiter
```

---

## Étape 2 — Transfert vers le CCRT

```bash
scp -r transfer_bundle/ user@ccrt-server:/workspace/
```

---

## Étape 3 — Sur le serveur CCRT

### 3.1 Créer l'environnement Python

```bash
python3.11 -m venv .venv
source .venv/bin/activate
```

### 3.2 Installer les dépendances depuis le bundle local

```bash
pip install --no-index --find-links=./pip_packages/ -r requirements.txt
```

### 3.3 Charger et démarrer Neo4j

```bash
docker load -i /workspace/neo4j-5.18.tar
docker run -d --name kg_neo4j -e NEO4J_AUTH=neo4j/nonenone -p 7474:7474 -p 7687:7687 -v neo4j_data:/data neo4j:5.18
```

### 3.4 Adapter config.py

Mettre à jour les chemins des modèles dans `config.py` :

```python
MODELS_DIR = "/workspace/models"

_PROFILES = {
    "local_server": {
        "extraction_model": f"{MODELS_DIR}/Qwen2.5-72B-Instruct-Q4_K_M.gguf",
        "answer_model":     f"{MODELS_DIR}/Qwen2.5-72B-Instruct-Q4_K_M.gguf",
        ...
    }
}
```

Et changer le MODE :

```python
MODE = "local_server"
```

### 3.5 Lancer le pipeline

```bash
cd /workspace/kg_pipeline
source .venv/bin/activate

python main.py --source /workspace/data/document.pdf
python main.py --interactive
```

---

## Checklist avant transfert

- [ ] GGUF téléchargé et vérifié
- [ ] `models/embeddings/bge-m3/` contient `config.json` et `tokenizer.json`
- [ ] `models/docling/` contient les deux sous-dossiers
- [ ] `pip_packages/` contient toutes les wheels
- [ ] `neo4j-5.18.tar` présent
- [ ] PDFs à traiter inclus dans `data/`
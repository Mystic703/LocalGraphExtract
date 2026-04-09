"""
prompts.py — All prompt templates in one place.
Edit here without touching pipeline logic.
"""

from llama_index.core.prompts import PromptTemplate

# ── Extraction prompt ──────────────────────────────────────────────────────────
# Deliberately schema-free so it works across diverse documents.
# Key design choices:
#   - No few-shot examples with concrete values (prevents hallucination anchoring)
#   - Explicit "verbatim only" instruction
#   - Hard cap on triplets + escape hatch (AUCUN_TRIPLET)

_EXTRACTION_SYSTEM = (
    "Tu es un expert en extraction d'information à partir de documents techniques.\n"
    "Tu dois extraire des triplets de connaissance UNIQUEMENT à partir du texte fourni.\n"
)

EXTRACTION_PROMPT = PromptTemplate(
    _EXTRACTION_SYSTEM
    + """
---------------------
Texte source :
{text}
---------------------

RÈGLES STRICTES :
1. Chaque mot dans le triplet DOIT être présent VERBATIM dans le texte source.
2. Ne génère PAS de valeurs numériques absentes du texte (températures, masses, taux...).
3. Si tu n'es pas certain qu'une valeur vient du texte, omets ce triplet.
4. Extrais entre 0 et 5 triplets maximum.
5. Si rien de pertinent n'est extractible : réponds uniquement AUCUN_TRIPLET.

Format STRICT (une ligne par triplet, rien d'autre) :
(Sujet, RELATION, Objet)

Triplets :"""
)


# ── Answer / RAG prompt ────────────────────────────────────────────────────────

EXPERT_PROMPT = PromptTemplate(
    "Tu es un expert en sûreté nucléaire.\n"
    "Voici les extraits EXACTS du rapport technique :\n"
    "---------------------\n"
    "{context_str}\n"
    "---------------------\n"
    "RÈGLES ABSOLUES :\n"
    "- Réponds UNIQUEMENT avec les informations présentes dans les extraits ci-dessus.\n"
    "- Ne répète JAMAIS deux fois la même phrase.\n"
    "- Si l'information n'est pas dans les extraits, dis 'Information non trouvée dans le rapport'.\n"
    "- Cite les valeurs numériques exactes (températures, durées, épaisseurs).\n\n"
    "Question : {query_str}\n\n"
    "Réponse (en français, sans répétition) :"
)

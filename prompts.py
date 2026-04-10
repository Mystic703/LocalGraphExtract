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
    _EXTRACTION_SYSTEM + """

INSTRUCTIONS (ne pas extraire ces lignes) :
1. Extrais UNIQUEMENT les faits présents dans le TEXTE SOURCE ci-dessous.
2. Les termes doivent apparaître TELS QUELS dans le texte source.
3. Ne génère PAS de valeurs numériques absentes du texte.
4. Maximum 5 triplets par texte.
5. Si rien n'est extractible : réponds uniquement AUCUN_TRIPLET

=== TEXTE SOURCE (extraire uniquement d'ici) ===
{text}
=== FIN DU TEXTE SOURCE ===

Format (une ligne par triplet, utilise des crochets) :
[terme_du_texte] -- RELATION --> [terme_du_texte]

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

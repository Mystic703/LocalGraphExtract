"""
prompts.py — All prompt templates in one place.
Edit here without touching pipeline logic.
"""

from llama_index.core.prompts import PromptTemplate

# ── Extraction prompt ──────────────────────────────────────────────────────────

_SYSTEM_PROMPT = """Tu es un expert en ingénierie système et en sûreté nucléaire.
Ton rôle est d'extraire des triplets de connaissance à partir de documents techniques."""

EXTRACTION_PROMPT = PromptTemplate(
    _SYSTEM_PROMPT + """

INSTRUCTIONS (ne pas extraire ces lignes) :
1. Extrais UNIQUEMENT les faits présents dans le TEXTE SOURCE ci-dessous.
2. Sujet ET Objet doivent apparaître TELS QUELS dans le texte source.
3. Ne génère PAS de valeurs numériques absentes du texte.
4. Si rien n'est extractible : réponds uniquement AUCUN_TRIPLET

=== TEXTE SOURCE (extraire uniquement d'ici) ===
{text}
=== FIN DU TEXTE SOURCE ===

Format (une ligne par triplet) :
(sujet_exact_du_texte, RELATION, objet_exact_du_texte)

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

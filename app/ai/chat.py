"""Chatbot de direction : répond en langage naturel à des questions sur l'activité du
magasin (ventes, stock, mouvements...), à partir d'un digest JSON préparé côté frontend
(voir buildAssistantContext() dans direction.js) et relayé tel quel par le backend Go.
Réutilise le client Anthropic mis en cache par claude_vision.py plutôt que d'en recréer un.
"""

import json
import logging
from typing import Any

from app.ai.claude_vision import _get_client, _supports_temperature
from app.core.config import settings

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "Tu t'appelles Lunette, l'assistant de direction de La Lunetterie, une chaîne de "
    "magasins d'optique. Tu réponds en français, de façon claire et concise, aux questions "
    "du gérant sur l'activité du magasin.\n\n"
    "Règles strictes :\n"
    "- Base-toi UNIQUEMENT sur les données JSON fournies ci-dessous. N'invente jamais un "
    "chiffre qui n'y figure pas.\n"
    "- Si la question porte sur une donnée absente de ce digest (chiffre d'affaires détaillé, "
    "comptabilité, réclamations, plannings...), dis clairement que ce module n'est pas encore "
    "disponible plutôt que de deviner.\n"
    "- Le digest contient la base complète (pas un échantillon) : `toutes_les_montures` liste "
    "CHAQUE monture, tous statuts confondus (en stock, en transit, réservée, vendue, perdue, "
    "cassée, retournée... voir son champ `status`), avec ses attributs (forme, couleur, "
    "matière, genre, marque, référence, prix, magasin) ; `mouvements` liste les déplacements. "
    "Pour une recherche précise (ex. \"combien de montures rondes en stock ?\", \"quelles "
    "montures de la marque X sont perdues ?\"), filtre/compte toi-même dans ces listes en te "
    "basant sur `status` plutôt que de te limiter aux compteurs déjà agrégés "
    "(`stock_actuel.par_categorie`, `ventes_par_categorie`, `*_par_jour`), qui ne sont là que "
    "comme raccourcis pour les questions globales et ne couvrent que le stock actif ou les "
    "ventes.\n"
    "- Réponds en phrases naturelles, SANS mise en forme markdown : pas d'astérisques, pas de "
    "listes à puces, pas de titres, pas de tableaux. Du texte parlé normal, avec des chiffres "
    "en toutes lettres ou en chiffres mais jamais de symboles décoratifs — tes réponses sont "
    "aussi lues à voix haute par une synthèse vocale qui prononcerait ces symboles.\n"
    "- Si l'utilisateur te demande explicitement d'ouvrir/afficher/aller sur une page ou un "
    "module (ex. \"ouvre le suivi des employés\", \"va sur les commandes fournisseur\", "
    "\"montre-moi l'historique\"), utilise l'outil navigate_to_page. Ne l'utilise PAS pour de "
    "simples questions sur ces sujets (ex. \"combien d'employés ?\" n'ouvre rien, ça répond "
    "juste avec les données) — uniquement pour une vraie demande de navigation. Si plusieurs "
    "pages sont demandées dans le même message, appelle l'outil une fois par page, dans "
    "l'ordre où l'utilisateur les a citées : l'interface les affichera l'une après l'autre. "
    "Après l'appel (ou les appels), confirme en une courte phrase.\n\n"
    "Données disponibles (JSON) :\n{context_json}"
)

# Pages du menu de direction.html (voir la constante MODULES dans direction.js) —
# tenue à jour manuellement en miroir de cette liste côté frontend.
NAVIGATE_TOOL = {
    "name": "navigate_to_page",
    "description": (
        "Ouvre une page/module du tableau de bord de direction à la place de l'utilisateur, "
        "uniquement quand il demande explicitement d'ouvrir/afficher/aller sur ce module."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "page": {
                "type": "string",
                "enum": [
                    "lunettes", "enregistrement", "ca", "employes", "paiements", "commandes",
                    "fournisseur", "compta", "planning", "reclamations", "messagerie", "historique",
                ],
                "description": "Identifiant de la page à ouvrir",
            }
        },
        "required": ["page"],
    },
}


def chat_reply(
    message: str, history: list[dict[str, Any]], context: dict[str, Any]
) -> tuple[str, list[dict[str, Any]]]:
    client = _get_client()
    if client is None:
        raise RuntimeError("ANTHROPIC_API_KEY non configurée")

    model = settings.ANTHROPIC_MODEL
    system_prompt = SYSTEM_PROMPT.format(context_json=json.dumps(context, ensure_ascii=False))

    messages: list[dict[str, Any]] = [{"role": m["role"], "content": m["content"]} for m in history]
    messages.append({"role": "user", "content": message})

    create_kwargs: dict[str, Any] = {
        "model": model,
        "system": system_prompt,
        "max_tokens": 1024,
        "messages": messages,
        "tools": [NAVIGATE_TOOL],
    }
    if _supports_temperature(model):
        create_kwargs["temperature"] = 0.3

    response = client.messages.create(**create_kwargs)

    actions: list[dict[str, Any]] = []
    tool_use_blocks = [b for b in response.content if getattr(b, "type", None) == "tool_use"]
    if tool_use_blocks:
        # Une action par tool_use, dans l'ordre d'appel : le frontend les ouvre l'une après
        # l'autre. L'API exige un tool_result pour CHAQUE tool_use de la réponse, sans quoi
        # l'appel suivant est rejeté avec une 400.
        navigate_blocks = [b for b in tool_use_blocks if b.name == "navigate_to_page"]
        actions = [{"type": "navigate", "page": b.input.get("page")} for b in navigate_blocks]

        messages.append({"role": "assistant", "content": response.content})
        messages.append(
            {
                "role": "user",
                "content": [
                    {"type": "tool_result", "tool_use_id": b.id, "content": "Page ouverte côté interface."}
                    for b in tool_use_blocks
                ],
            }
        )
        follow_up_kwargs = dict(create_kwargs)
        follow_up_kwargs["messages"] = messages
        response = client.messages.create(**follow_up_kwargs)

    reply = "".join(block.text for block in response.content if getattr(block, "type", None) == "text")
    logger.info("Réponse chat direction: %s (actions=%s)", reply[:200], actions)
    return reply.strip(), actions

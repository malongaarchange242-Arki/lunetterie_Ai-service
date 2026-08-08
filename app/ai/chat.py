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
    "- `mouvements` porte une date ET une heure (ISO complet), et `maintenant` te donne "
    "l'horodatage courant : tu peux donc répondre aux questions de période fine "
    "(\"l'activité de ce matin\", \"ce qui est sorti depuis midi\", \"hier après-midi\") en "
    "comparant toi-même les horodatages. Ne réponds pas que tu n'as pas le temps réel : le "
    "digest est reconstruit à chaque ouverture du chat.\n"
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
    "Après l'appel (ou les appels), confirme en une courte phrase.\n"
    "- Si l'utilisateur cherche une monture pour un client avec des caractéristiques "
    "(genre, forme, gamme, taille) — ex. \"une ovale pour femme en 52\", \"tu as du luxe "
    "homme ?\" —, appelle l'outil rechercher_monture EN PLUS de ta réponse, que la monture "
    "soit trouvée en stock ou non : c'est ce qui alimente le panier de demande du magasin. "
    "Ne renseigne que les critères réellement exprimés, laisse les autres vides plutôt que "
    "de les deviner. Si la ville n'est pas dite, prends celle du contexte "
    "(`ville_courante`). N'appelle PAS cet outil pour une question statistique globale "
    "(\"combien de montures rondes en stock ?\") : là il s'agit de compter, pas de "
    "chercher pour un client.\n\n"
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

# Recherche de monture pour un client : chaque appel dépose une ligne dans le panier de
# demande du magasin (table demand_baskets côté Go). Les énumérations reprennent
# exactement les valeurs stockées en base — voir normalizeShapeName/resolveFrameGamme
# côté React, qui servent ensuite à rapprocher la demande du stock principal.
SEARCH_GLASSES_TOOL = {
    "name": "rechercher_monture",
    "description": (
        "Enregistre la recherche d'une monture pour un client d'un magasin donné. À appeler "
        "dès que l'utilisateur cherche une monture avec des caractéristiques, qu'elle soit "
        "trouvée en stock ou non. Ne pas appeler pour une question statistique sur le stock."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "ville": {
                "type": "string",
                "description": "Ville du magasin concerné (ex. Pointe-Noire, Kinshasa)",
            },
            "genre": {
                "type": "string",
                "enum": ["Homme", "Femme", "Enfant", "Unisexe"],
                "description": "Genre demandé, seulement s'il est exprimé",
            },
            "forme": {
                "type": "string",
                "enum": [
                    "Rectangle", "Rond", "Ovale", "Carré", "Papillon", "Aviateur", "Wayfarer",
                    "Cat-eye", "Clubmaster", "Browline", "Hexagonal", "Pantos", "Masque",
                ],
                "description": "Forme demandée, seulement si elle est exprimée",
            },
            "gamme": {
                "type": "string",
                "enum": ["classique", "moyenne", "luxe"],
                "description": "Gamme de prix demandée, seulement si elle est exprimée",
            },
            "taille": {
                "type": "string",
                "description": "Taille de la monture si précisée (ex. 52, 54)",
            },
        },
        "required": ["ville"],
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
        "tools": [NAVIGATE_TOOL, SEARCH_GLASSES_TOOL],
    }
    if _supports_temperature(model):
        create_kwargs["temperature"] = 0.3

    response = client.messages.create(**create_kwargs)

    actions: list[dict[str, Any]] = []
    tool_use_blocks = [b for b in response.content if getattr(b, "type", None) == "tool_use"]
    if tool_use_blocks:
        # Une action par tool_use, dans l'ordre d'appel : le frontend les traite l'une après
        # l'autre. L'API exige un tool_result pour CHAQUE tool_use de la réponse, sans quoi
        # l'appel suivant est rejeté avec une 400 — d'où la boucle unique qui construit
        # action et tool_result ensemble.
        tool_results: list[dict[str, Any]] = []
        for block in tool_use_blocks:
            if block.name == "navigate_to_page":
                actions.append({"type": "navigate", "page": block.input.get("page") or ""})
                result = "Page ouverte côté interface."
            elif block.name == "rechercher_monture":
                actions.append(
                    {
                        "type": "search",
                        "ville": block.input.get("ville") or "",
                        "genre": block.input.get("genre") or "",
                        "forme": block.input.get("forme") or "",
                        "gamme": block.input.get("gamme") or "",
                        "taille": block.input.get("taille") or "",
                    }
                )
                result = "Recherche déposée dans le panier de demande du magasin."
            else:
                logger.warning("Outil inconnu appelé par le chatbot: %s", block.name)
                result = "Outil inconnu, ignoré."
            tool_results.append({"type": "tool_result", "tool_use_id": block.id, "content": result})

        messages.append({"role": "assistant", "content": response.content})
        messages.append({"role": "user", "content": tool_results})
        follow_up_kwargs = dict(create_kwargs)
        follow_up_kwargs["messages"] = messages
        response = client.messages.create(**follow_up_kwargs)

    reply = "".join(block.text for block in response.content if getattr(block, "type", None) == "text")
    logger.info("Réponse chat direction: %s (actions=%s)", reply[:200], actions)
    return reply.strip(), actions

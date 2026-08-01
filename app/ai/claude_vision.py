"""Analyse de montures via l'API Claude (vision) : détection visuelle (forme, couleur,
matière, genre) sur la photo de face, et OCR de la référence/marque sur la photo de la
branche. Utilisé en complément (et non en remplacement définitif) du pipeline YOLO local :
si la clé API est absente ou l'appel échoue, l'appelant retombe sur les résultats locaux.
"""

import base64
import json
import logging
import mimetypes
from pathlib import Path
from typing import Any

from app.core.config import settings

logger = logging.getLogger(__name__)

SHAPES = ["Aviateur", "Rond", "Ovale", "Carré", "Rectangulaire", "Papillon", "Oeil de chat", "Sport", "Wayfarer"]
COLORS = ["Noir", "Marron", "Bleu", "Rouge", "Vert", "Gris", "Blanc", "Doré", "Argenté", "Violet"]
MATERIALS = ["Acétate", "Métal", "Plastique", "Titane", "Bois", "Composite", "Inox"]
GENDERS = ["Homme", "Femme", "Enfant", "Unisexe"]

_client = None


def _get_client():
    global _client
    if _client is not None:
        return _client
    if not settings.ANTHROPIC_API_KEY:
        return None
    import anthropic

    _client = anthropic.Anthropic(api_key=settings.ANTHROPIC_API_KEY)
    return _client


def _encode_image(image_path: str) -> tuple[str, str]:
    media_type = mimetypes.guess_type(image_path)[0] or "image/jpeg"
    data = Path(image_path).read_bytes()
    return base64.standard_b64encode(data).decode("ascii"), media_type


def _parse_json_reply(text: str) -> dict[str, Any] | None:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        if cleaned.startswith("json"):
            cleaned = cleaned[4:]
    try:
        return json.loads(cleaned.strip())
    except (json.JSONDecodeError, ValueError):
        logger.warning("Réponse Claude non-JSON: %s", text[:200])
        return None


def _call_claude(image_path: str, prompt: str) -> dict[str, Any] | None:
    client = _get_client()
    if client is None:
        return None

    try:
        data, media_type = _encode_image(image_path)
        message = client.messages.create(
            model=settings.ANTHROPIC_MODEL,
            max_tokens=512,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "source": {"type": "base64", "media_type": media_type, "data": data}},
                        {"type": "text", "text": prompt},
                    ],
                }
            ],
        )
        text = "".join(block.text for block in message.content if getattr(block, "type", None) == "text")
        logger.info("Réponse brute Claude vision: %s", text[:500])
        return _parse_json_reply(text)
    except Exception as exc:  # défensif: jamais bloquant pour l'appelant
        logger.warning("Appel Claude vision échoué: %s", exc)
        return None


def analyze_monture(image_path: str) -> dict[str, Any] | None:
    """Détecte forme, couleur, matière, genre et marque depuis la photo de face de la monture.
    La marque est souvent imprimée/gravée sur le verre ou la face (pas seulement sur la
    branche) : ex. "Charlie Duke" imprimé sur le verre droit."""
    prompt = (
        "Tu analyses la photo d'une monture de lunettes pour un inventaire optique. "
        f"Réponds UNIQUEMENT avec un objet JSON strict (pas de texte autour), avec exactement ces clés :\n"
        f'"shape" (une valeur EXACTE parmi {SHAPES}),\n'
        f'"color" (une valeur EXACTE parmi {COLORS}, la couleur dominante de la monture),\n'
        f'"material" (une valeur EXACTE parmi {MATERIALS}),\n'
        f'"gender" (une valeur EXACTE parmi {GENDERS}, le style visé par la monture),\n'
        '"brand" (le nom de marque lu s\'il est imprimé/gravé sur le verre ou la face, ex: "Charlie Duke", "Ray-Ban" — ou null si absent/illisible, n\'invente rien),\n'
        '"confidence" (nombre entre 0 et 1, ta confiance globale).\n'
        "Si une caractéristique est vraiment indéterminable, mets null pour cette clé."
    )
    result = _call_claude(image_path, prompt)
    if not result:
        return None

    def _valid(value: Any, allowed: list[str]) -> str | None:
        return value if isinstance(value, str) and value in allowed else None

    def _clean(value: Any) -> str | None:
        if isinstance(value, str) and value.strip():
            return value.strip()
        return None

    return {
        "shape": _valid(result.get("shape"), SHAPES),
        "color": _valid(result.get("color"), COLORS),
        "material": _valid(result.get("material"), MATERIALS),
        "gender": _valid(result.get("gender"), GENDERS),
        "brand": _clean(result.get("brand")),
        "confidence": float(result.get("confidence") or 0.85),
    }


def ocr_branche(image_path: str) -> dict[str, Any] | None:
    """Lit par OCR la référence et la marque gravées/imprimées sur la branche de la monture."""
    prompt = (
        "Tu regardes la photo de la branche (temple) d'une monture de lunettes. Le texte gravé "
        "ou imprimé dessus contient généralement la marque et une référence modèle (souvent un "
        "code alphanumérique, ex: RB2180-001, parfois avec une taille comme 55□18-145).\n"
        "Réponds UNIQUEMENT avec un objet JSON strict, avec exactement ces clés :\n"
        '"reference" (le code de référence lu tel quel, ou null si illisible/absent),\n'
        '"brand" (le nom de la marque lue, ou null si illisible/absente),\n'
        '"confidence" (nombre entre 0 et 1).\n'
        "N'invente rien : si le texte n'est pas net, mets null plutôt que de deviner."
    )
    result = _call_claude(image_path, prompt)
    if not result:
        return None

    def _clean(value: Any) -> str | None:
        if isinstance(value, str) and value.strip():
            return value.strip()
        return None

    return {
        "reference": _clean(result.get("reference")),
        "brand": _clean(result.get("brand")),
        "confidence": float(result.get("confidence") or 0.7),
    }

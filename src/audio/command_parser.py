from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

from src.models.events import VoiceIntent


@dataclass(frozen=True)
class ParsedCommand:
    """Resultado determinístico del analizador de catálogo acotado de voz."""

    intent: VoiceIntent
    concept: str | None = None
    reason: str = ""


# Palabras vacías o fragmentos que NO deben aceptarse como concepto de un nodo
_INVALID_CONCEPTS = {
    "no",
    "si",
    "sí",
    "un",
    "una",
    "el",
    "la",
    "los",
    "las",
    "de",
    "del",
    "y",
    "o",
    "a",
    "en",
    "con",
    "por",
    "para",
    "nodo",
    "nuevo",
    "crear",
}


def normalize_spanish(text: str) -> str:
    """Normaliza a minúsculas sin signos de puntuación preservando letras españolas."""
    cleaned = text.strip().lower()
    cleaned = re.sub(r"[^\w\sáéíóúüñ]", " ", cleaned, flags=re.UNICODE)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def strip_accents(text: str) -> str:
    nfkd = unicodedata.normalize("NFKD", text)
    return "".join(ch for ch in nfkd if not unicodedata.combining(ch))


def parse_voice_command(raw_text: str) -> ParsedCommand:
    """Analiza texto ASR contra el catálogo acotado.

    Catálogo permitido:
    - "Crear nodo [Concepto/Palabra]"
    - "Conectar" (y variantes explícitas "conectar nodos" / "conectar nodo")
    - "Borrar" / "Eliminar" (y "borrar/eliminar nodo/enlace")
    - "Deshacer" (y "deshacer accion")

    Cualquier frase trunca (ej. "crear", "crear no", "crear nodo") o fuera de catálogo
    se clasifica como AMBIGUOUS o UNRECOGNIZED y nunca dispara cambios estructurales.
    """
    normalized = normalize_spanish(raw_text)
    if not normalized:
        return ParsedCommand(
            intent=VoiceIntent.UNRECOGNIZED,
            reason="Audio vacío o sin transcripción reconocible.",
        )

    ascii_norm = strip_accents(normalized)

    # 1. Crear nodo [Concepto]
    if ascii_norm.startswith("crear"):
        match = re.match(r"^crear\s+nodo(?:\s+(.+))?$", normalized, flags=re.UNICODE)
        if not match:
            return ParsedCommand(
                intent=VoiceIntent.AMBIGUOUS,
                reason=(
                    f"Comando incompleto '{raw_text.strip()}'. "
                    "Di exactamente: 'Crear nodo [concepto]'."
                ),
            )
        concept_raw = (match.group(1) or "").strip()
        if not concept_raw:
            return ParsedCommand(
                intent=VoiceIntent.AMBIGUOUS,
                reason="Falta el nombre del concepto tras 'Crear nodo'.",
            )
        concept_tokens = concept_raw.split()
        if len(concept_tokens) == 1 and strip_accents(concept_tokens[0]) in _INVALID_CONCEPTS:
            return ParsedCommand(
                intent=VoiceIntent.AMBIGUOUS,
                reason=f"El concepto '{concept_raw}' es ambiguo. Repite 'Crear nodo [concepto]'.",
            )
        # Capitalizar legiblemente preservando acentos originales
        concept_display = concept_raw[0].upper() + concept_raw[1:]
        return ParsedCommand(
            intent=VoiceIntent.CREATE_NODE,
            concept=concept_display,
            reason=f"Crear nodo '{concept_display}'",
        )

    # 2. Conectar
    if ascii_norm in {"conectar", "conectar nodo", "conectar nodos", "conectar enlace"}:
        return ParsedCommand(
            intent=VoiceIntent.CONNECT,
            reason="Modo conexión origen-destino activado.",
        )
    if ascii_norm.startswith("conectar "):
        return ParsedCommand(
            intent=VoiceIntent.AMBIGUOUS,
            reason="Di únicamente 'Conectar' tras seleccionar el nodo origen.",
        )

    # 3. Borrar / Eliminar
    if ascii_norm in {
        "borrar",
        "eliminar",
        "borrar nodo",
        "eliminar nodo",
        "borrar enlace",
        "eliminar enlace",
        "borrar elemento",
        "eliminar elemento",
    }:
        return ParsedCommand(
            intent=VoiceIntent.DELETE,
            reason="Solicitud de eliminación del elemento seleccionado.",
        )

    # 4. Deshacer
    if ascii_norm in {"deshacer", "deshacer accion", "deshacer ultimo", "deshacer ultima"}:
        return ParsedCommand(
            intent=VoiceIntent.UNDO,
            reason="Revertir última operación compatible.",
        )

    return ParsedCommand(
        intent=VoiceIntent.UNRECOGNIZED,
        reason=(
            f"Comando no reconocido: '{raw_text.strip()}'. "
            "Usa: 'Crear nodo [concepto]', 'Conectar', 'Borrar' o 'Deshacer'."
        ),
    )

from __future__ import annotations

import pytest

from src.audio.command_parser import parse_voice_command
from src.models.events import VoiceIntent


@pytest.mark.parametrize(
    "raw_text, expected_concept",
    [
        ("Crear nodo Percepción", "Percepción"),
        ("crear nodo memoria de trabajo", "Memoria de trabajo"),
        ("  Crear   nodo   Atención! ", "Atención"),
    ],
)
def test_valid_create_node_commands(raw_text: str, expected_concept: str) -> None:
    parsed = parse_voice_command(raw_text)
    assert parsed.intent == VoiceIntent.CREATE_NODE
    assert parsed.concept == expected_concept


@pytest.mark.parametrize(
    "raw_text",
    [
        "crear",          # Regresión directa de PART-2026-020_log.json
        "crear no",       # Regresión directa de PART-2026-020_log.json
        "crear nodo",
        "crear nodo no",
        "crear nodo el",
    ],
)
def test_ambiguous_create_commands_never_create_nodes(raw_text: str) -> None:
    parsed = parse_voice_command(raw_text)
    assert parsed.intent == VoiceIntent.AMBIGUOUS
    assert parsed.concept is None


def test_bounded_catalog_other_commands() -> None:
    assert parse_voice_command("Conectar").intent == VoiceIntent.CONNECT
    assert parse_voice_command("conectar nodos").intent == VoiceIntent.CONNECT
    assert parse_voice_command("Borrar").intent == VoiceIntent.DELETE
    assert parse_voice_command("Eliminar").intent == VoiceIntent.DELETE
    assert parse_voice_command("eliminar nodo").intent == VoiceIntent.DELETE
    assert parse_voice_command("Deshacer").intent == VoiceIntent.UNDO
    assert parse_voice_command("deshacer acción").intent == VoiceIntent.UNDO


def test_unrecognized_phrases_rejected_cleanly() -> None:
    assert parse_voice_command("").intent == VoiceIntent.UNRECOGNIZED
    assert parse_voice_command("hola buenos días").intent == VoiceIntent.UNRECOGNIZED
    assert parse_voice_command("mover hacia arriba").intent == VoiceIntent.UNRECOGNIZED

"""Carrega a chave AES-256 compartilhada a partir da variável ENCRYPTION_KEY."""
import base64
import os


def get_key() -> bytes:
    raw = os.environ.get("ENCRYPTION_KEY", "")
    if not raw:
        raise RuntimeError("Defina ENCRYPTION_KEY no arquivo .env.")
    try:
        key = base64.b64decode(raw, validate=True)
    except Exception as exc:
        raise RuntimeError("ENCRYPTION_KEY deve estar em base64.") from exc
    if len(key) != 32:
        raise RuntimeError("ENCRYPTION_KEY deve ter exatamente 32 bytes (AES-256).")
    return key

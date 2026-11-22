"""Descriptografia (AES-256-GCM) e descompactação (gzip) de bytes."""
import gzip

from cryptography.hazmat.primitives.ciphers.aead import AESGCM

from .keys import get_key

NONCE_SIZE = 12


def decrypt_bytes(blob: bytes) -> bytes:
    if len(blob) <= NONCE_SIZE:
        raise ValueError("Conteúdo criptografado inválido.")
    nonce, ciphertext = blob[:NONCE_SIZE], blob[NONCE_SIZE:]
    return AESGCM(get_key()).decrypt(nonce, ciphertext, None)


def decompress_bytes(data: bytes) -> bytes:
    return gzip.decompress(data)


def decrypt_and_decompress(blob: bytes) -> bytes:
    return decompress_bytes(decrypt_bytes(blob))

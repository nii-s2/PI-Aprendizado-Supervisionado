"""Compactação (gzip) e criptografia (AES-256-GCM) de bytes.

Formato do blob criptografado:  nonce (12 bytes) || ciphertext+tag
O GCM autentica os dados: qualquer alteração faz a descriptografia falhar.
"""
import gzip
import os

from cryptography.hazmat.primitives.ciphers.aead import AESGCM

from .keys import get_key

NONCE_SIZE = 12


def compress_bytes(data: bytes) -> bytes:
    return gzip.compress(data, compresslevel=6)


def encrypt_bytes(data: bytes) -> bytes:
    nonce = os.urandom(NONCE_SIZE)
    return nonce + AESGCM(get_key()).encrypt(nonce, data, None)


def compress_and_encrypt(data: bytes) -> bytes:
    """Primeiro compacta, depois criptografa (dado criptografado não comprime)."""
    return encrypt_bytes(compress_bytes(data))

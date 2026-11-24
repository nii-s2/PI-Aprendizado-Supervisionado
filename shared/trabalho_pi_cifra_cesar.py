"""Tratamento ponta-a-ponta dos arquivos: zlib + cifra de César + base64."""
import base64
import zlib

NOME_API_CIFRA_CESAR = "trabalho_pi_cifra_cesar"


def cifra_cesar_bytes(dados: bytes, deslocamento: int, cifrar: bool = True) -> bytes:
    passo = (deslocamento % 256) if cifrar else -(deslocamento % 256)
    return bytes((byte + passo) % 256 for byte in dados)


def empacotar_cesar(csv_bytes: bytes, deslocamento: int = 7) -> str:
    comprimido = zlib.compress(csv_bytes, level=9)
    cifrado = cifra_cesar_bytes(comprimido, deslocamento, cifrar=True)
    return base64.b64encode(cifrado).decode("ascii")


def desempacotar_cesar(pacote_b64: str, deslocamento: int = 7) -> bytes:
    cifrado = base64.b64decode(pacote_b64.encode("ascii"))
    comprimido = cifra_cesar_bytes(cifrado, deslocamento, cifrar=False)
    return zlib.decompress(comprimido)

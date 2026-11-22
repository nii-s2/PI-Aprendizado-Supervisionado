"""API do servidor: recebe bases criptografadas, treina, testa e devolve os resultados."""
import base64
import io
import logging

import pandas as pd
from flask import Flask, jsonify, request

from shared.compress_encrypt import compress_and_encrypt, compress_bytes, encrypt_bytes
from shared.decompress_decrypt import decrypt_and_decompress
from model import run_regression

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 100 * 1024 * 1024  # 100 MB
log = logging.getLogger("server")


def _load_csv(field: str, nome: str) -> pd.DataFrame:
    f = request.files.get(field)
    if f is None:
        raise ValueError(f"Base de {nome} não enviada.")
    try:
        raw = decrypt_and_decompress(f.read())
    except Exception as exc:
        raise ValueError(f"Não foi possível descriptografar a base de {nome}. Confira a chave.") from exc
    try:
        return pd.read_csv(io.BytesIO(raw))
    except Exception as exc:
        raise ValueError(f"A base de {nome} não é um CSV válido.") from exc


def _b64(data: bytes) -> str:
    return base64.b64encode(data).decode()


@app.get("/health")
def health():
    return jsonify(status="ok")


@app.post("/train")
def train():
    try:
        train_df = _load_csv("train", "treino")
        test_df = _load_csv("test", "teste")
        metrics, results, plot_png = run_regression(train_df, test_df, request.form.get("target"))

        csv_gz = compress_bytes(results.to_csv(index=False).encode("utf-8"))
        return jsonify(
            metrics=metrics,
            results_csv_gz_enc=_b64(encrypt_bytes(csv_gz)),   # CSV compactado, depois criptografado
            plot_png_enc=_b64(compress_and_encrypt(plot_png)),  # PNG compactado e criptografado
        )
    except ValueError as exc:
        return jsonify(error=str(exc)), 400
    except Exception:
        log.exception("Erro inesperado")
        return jsonify(error="Erro interno ao treinar o modelo."), 500

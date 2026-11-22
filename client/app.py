"""Client: serve o site e, a cada envio, compacta + criptografa as bases antes de
mandá-las ao server; depois descriptografa os resultados para o navegador."""
import base64
import os

import requests
from flask import Flask, jsonify, request, send_from_directory

from shared.compress_encrypt import compress_and_encrypt
from shared.decompress_decrypt import decrypt_and_decompress, decrypt_bytes

SERVER_URL = os.environ.get("SERVER_URL", "http://server:5000")
WEB_DIR = os.path.join(os.path.dirname(__file__), "web")

app = Flask(__name__, static_folder=WEB_DIR, static_url_path="")
app.config["MAX_CONTENT_LENGTH"] = 100 * 1024 * 1024


@app.get("/")
def index():
    return send_from_directory(WEB_DIR, "index.html")


def _read_upload(field: str, nome: str) -> bytes:
    f = request.files.get(field)
    if f is None or not f.filename:
        raise ValueError(f"Envie a base de {nome}.")
    if not f.filename.lower().endswith(".csv"):
        raise ValueError(f"A base de {nome} deve ser um arquivo .csv.")
    return f.read()


@app.post("/api/process")
def process():
    try:
        train = _read_upload("train", "treino")
        test = _read_upload("test", "teste")
    except ValueError as exc:
        return jsonify(error=str(exc)), 400

    try:
        resp = requests.post(
            f"{SERVER_URL}/train",
            files={
                "train": ("train.csv.gz.enc", compress_and_encrypt(train)),
                "test": ("test.csv.gz.enc", compress_and_encrypt(test)),
            },
            data={"target": request.form.get("target", "")},
            timeout=120,
        )
    except requests.RequestException:
        return jsonify(error="Não foi possível falar com o servidor de treino."), 502

    body = resp.json() if resp.headers.get("content-type", "").startswith("application/json") else {}
    if resp.status_code != 200:
        return jsonify(error=body.get("error", "Falha no servidor de treino.")), resp.status_code

    try:
        csv_gz = decrypt_bytes(base64.b64decode(body["results_csv_gz_enc"]))     # continua .gz
        plot_png = decrypt_and_decompress(base64.b64decode(body["plot_png_enc"]))
    except Exception:
        return jsonify(error="Não foi possível descriptografar a resposta do servidor."), 502

    return jsonify(
        metrics=body["metrics"],
        results_csv_gz=base64.b64encode(csv_gz).decode(),
        plot_png=base64.b64encode(plot_png).decode(),
    )

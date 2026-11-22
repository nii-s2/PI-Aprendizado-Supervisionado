"""Client local: serve o site HTML e usa shared/trabalho_pi_cifra_cesar.py para
compactar/criptografar antes de chamar o server local e descompactar/descriptografar na volta."""
import base64
import io
import os
import sys
import zipfile

import requests
from flask import Flask, jsonify, request, send_from_directory

CLIENT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(CLIENT_DIR, ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from shared.compress_encrypt import empacotar_cesar
from shared.decompress_decrypt import desempacotar_cesar

SERVER_URL = os.environ.get("SERVER_URL", "http://127.0.0.1:5000")
WEB_DIR = os.path.join(CLIENT_DIR, "web")

app = Flask(__name__, static_folder=WEB_DIR, static_url_path="")
app.config["MAX_CONTENT_LENGTH"] = 100 * 1024 * 1024


class ErroApi(Exception):
    def __init__(self, mensagem, codigo=400):
        super().__init__(mensagem)
        self.codigo = codigo


@app.errorhandler(ErroApi)
def _erro(exc):
    return jsonify(error=str(exc)), exc.codigo


@app.get("/")
def index():
    return send_from_directory(WEB_DIR, "index.html")


def _chave() -> int:
    try:
        return int(request.form.get("chave", "").strip())
    except ValueError:
        raise ErroApi("A chave de criptografia deve ser um número inteiro.")


def _csv(campo: str, nome: str) -> bytes:
    f = request.files.get(campo)
    if f is None or not f.filename:
        raise ErroApi(f"Envie a base de {nome}.")
    if not f.filename.lower().endswith(".csv"):
        raise ErroApi(f"A base de {nome} deve ser um arquivo .csv.")
    return f.read()


def _server(metodo: str, rota: str, payload: dict | None = None) -> dict:
    try:
        resp = requests.request(metodo, f"{SERVER_URL}{rota}", json=payload, timeout=300)
    except requests.RequestException:
        raise ErroApi(
            f"Não foi possível falar com o servidor local ({SERVER_URL}). Verifique se server/app.py está rodando.",
            502,
        )
    try:
        corpo = resp.json()
    except ValueError:
        corpo = {}
    if resp.status_code != 200:
        raise ErroApi(corpo.get("erro", "Falha no servidor de treino."), resp.status_code)
    return corpo


@app.get("/api/status")
def status():
    return jsonify(_server("GET", "/status"))


@app.post("/api/reset")
def reset():
    return jsonify(_server("POST", "/reset", {}))


@app.post("/api/treinar")
def treinar():
    chave = _chave()
    csv_bytes = _csv("train", "treino")
    pacote = empacotar_cesar(csv_bytes, deslocamento=chave)   # zlib + César + base64
    corpo = _server("POST", "/treinar", {"pacote_b64": pacote, "deslocamento": chave})

    pacote_grafico = corpo.pop("pacote_grafico_treino_b64", None)
    if pacote_grafico:
        try:
            png = desempacotar_cesar(pacote_grafico, deslocamento=chave)
        except Exception:
            raise ErroApi("Não foi possível descriptografar o gráfico de treino.", 502)
        corpo["grafico_treino_png"] = base64.b64encode(png).decode()
    return jsonify(corpo)


@app.post("/api/testar")
def testar():
    chave = _chave()
    csv_bytes = _csv("test", "teste")
    pacote = empacotar_cesar(csv_bytes, deslocamento=chave)
    corpo = _server("POST", "/testar", {"pacote_b64": pacote, "deslocamento": chave})

    try:
        csv_resultado = desempacotar_cesar(corpo["pacote_resposta_b64"], deslocamento=chave)
        grafico_png = desempacotar_cesar(corpo["pacote_grafico_b64"], deslocamento=chave)
    except Exception:
        raise ErroApi("Não foi possível descriptografar a resposta do servidor.", 502)

    zip_buf = io.BytesIO()                                    # CSV de resultado compactado
    with zipfile.ZipFile(zip_buf, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("previsoes_time.csv", csv_resultado)

    return jsonify(
        avaliacao=corpo["avaliacao"],
        resultado_zip=base64.b64encode(zip_buf.getvalue()).decode(),
        grafico_png=base64.b64encode(grafico_png).decode(),
    )


if __name__ == "__main__":
    host = os.environ.get("CLIENT_HOST", "127.0.0.1")
    port = int(os.environ.get("CLIENT_PORT", "8080"))
    print(f"Interface Web (Client) rodando localmente em http://{host}:{port}")
    app.run(host=host, port=port, debug=False, threaded=True)

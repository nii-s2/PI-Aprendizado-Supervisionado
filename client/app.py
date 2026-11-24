"""Client local: serve o site HTML e usa shared/trabalho_pi_cifra_cesar.py para
compactar/criptografar antes de chamar o server local e descompactar/descriptografar na volta."""
import base64
import io
import logging
import os
import sys
import threading
import time
import zipfile

import requests
from flask import Flask, jsonify, request, send_from_directory

CLIENT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(CLIENT_DIR, ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from shared.compress_encrypt import empacotar_cesar
from shared.decompress_decrypt import desempacotar_cesar

SERVER_URL = os.environ.get("SERVER_URL", "http://127.0.0.1:5000").rstrip("/")
API_TOKEN = os.environ.get("API_TOKEN", "")
HEADERS = {"X-API-Token": API_TOKEN} if API_TOKEN else {}
# Plano gratuito do Render dorme após 15 min sem uso e leva ~1 min para acordar.
# Localmente o padrão é 3 tentativas (rápido); no Render use WAKE_TENTATIVAS=12.
WAKE_TENTATIVAS = int(os.environ.get("WAKE_TENTATIVAS", "3"))
WEB_DIR = os.path.join(CLIENT_DIR, "web")

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("client")

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


def _acordar_servidor(tentativas: int = WAKE_TENTATIVAS, espera: float = 5.0) -> str | None:
    """Chama /health até o servidor responder (acorda o Render gratuito).
    Devolve None se respondeu, ou o motivo da última falha."""
    motivo = "sem resposta"
    for i in range(tentativas):
        try:
            resp = requests.request("GET", f"{SERVER_URL}/health", timeout=15)
            if resp.status_code == 200:
                return None
            motivo = f"{SERVER_URL}/health respondeu HTTP {resp.status_code}"
        except requests.RequestException as exc:
            motivo = f"falha de conexão com {SERVER_URL} ({type(exc).__name__})"
        log.warning("Servidor inacessível (tentativa %d/%d): %s", i + 1, tentativas, motivo)
        if i < tentativas - 1:
            time.sleep(espera)
    return motivo


# acorda o servidor em paralelo ao boot do client, para o 1º acesso não esperar duas vezes
if os.environ.get("WAKE_ON_START", "1") == "1":
    threading.Thread(target=_acordar_servidor, daemon=True).start()


def _server(metodo: str, rota: str, payload: dict | None = None) -> dict:
    motivo = _acordar_servidor()
    if motivo:
        raise ErroApi(
            f"Não foi possível falar com o servidor: {motivo}. Se ele está no plano gratuito do Render, "
            "pode estar iniciando (cerca de 1 minuto): tente de novo. Se persistir, confira se SERVER_URL "
            "é exatamente a URL do regressao-server. Em execução local, verifique se server/app.py está rodando.",
            503,
        )
    try:
        resp = requests.request(metodo, f"{SERVER_URL}{rota}", json=payload, headers=HEADERS, timeout=300)
    except requests.RequestException:
        raise ErroApi("A conexão com o servidor foi interrompida durante o processamento.", 502)
    try:
        corpo = resp.json()
    except ValueError:
        corpo = {}
    if resp.status_code == 401:
        raise ErroApi("O servidor recusou o token de acesso: confira se API_TOKEN é o mesmo no client e no server.", 502)
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
        pacote_grafico = corpo.get("pacote_grafico_b64")   # ausente quando o teste não tem rótulos
        grafico_png = desempacotar_cesar(pacote_grafico, deslocamento=chave) if pacote_grafico else None
    except Exception:
        raise ErroApi("Não foi possível descriptografar a resposta do servidor.", 502)

    zip_buf = io.BytesIO()                                    # CSV de resultado compactado
    with zipfile.ZipFile(zip_buf, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("previsoes_time.csv", csv_resultado)

    resposta = {
        "avaliacao": corpo["avaliacao"],
        "resultado_csv": base64.b64encode(csv_resultado).decode(),
        "resultado_zip": base64.b64encode(zip_buf.getvalue()).decode(),
    }
    if grafico_png:
        resposta["grafico_png"] = base64.b64encode(grafico_png).decode()
    return jsonify(resposta)


if __name__ == "__main__":
    host = os.environ.get("CLIENT_HOST", "127.0.0.1")
    port = int(os.environ.get("CLIENT_PORT", "8080"))
    print(f"Interface Web (Client) rodando localmente em http://{host}:{port}")
    app.run(host=host, port=port, debug=False, threaded=True)

"""API local do servidor: /health, /status, /reset, /treinar, /testar."""
import logging
import os
import sys
import threading

from flask import Flask, jsonify, request

SERVER_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(SERVER_DIR, ".."))
for p in (SERVER_DIR, ROOT_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)

from nucleo_pi import ServicoRegressaoPI

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 100 * 1024 * 1024
log = logging.getLogger("server")

SERVICO = ServicoRegressaoPI()
TRAVA = threading.Lock()  # um treino/teste/reset por vez


def _dados_pacote():
    dados = request.get_json(silent=True) or {}
    if "pacote_b64" not in dados:
        raise ValueError("Requisição sem 'pacote_b64'.")
    try:
        deslocamento = int(dados.get("deslocamento", 7))
    except (TypeError, ValueError) as exc:
        raise ValueError("'deslocamento' deve ser um número inteiro.") from exc
    return dados["pacote_b64"], deslocamento


def _executar(acao):
    try:
        with TRAVA:
            return jsonify(acao())
    except (ValueError, RuntimeError) as exc:
        return jsonify(erro=str(exc)), 400
    except Exception:
        log.exception("Erro inesperado")
        return jsonify(erro="Erro interno no servidor."), 500


@app.get("/health")
def health():
    return jsonify(status="ok")


@app.get("/status")
def status():
    return _executar(SERVICO.status)


@app.post("/reset")
def reset():
    def acao():
        SERVICO.resetar()
        return {"status": "resetado", "mensagem": "Modelo resetado com sucesso."}
    return _executar(acao)


@app.post("/treinar")
def treinar():
    def acao():
        pacote, desl = _dados_pacote()
        return SERVICO.treinar_pacote(pacote, desl)
    return _executar(acao)


@app.post("/testar")
def testar():
    def acao():
        pacote, desl = _dados_pacote()
        return SERVICO.testar_pacote(pacote, desl)
    return _executar(acao)


if __name__ == "__main__":
    host = os.environ.get("SERVER_HOST", "127.0.0.1")
    port = int(os.environ.get("SERVER_PORT", "5000"))
    print(f"Servidor de processamento rodando localmente em http://{host}:{port}")
    app.run(host=host, port=port, debug=False, threaded=True)

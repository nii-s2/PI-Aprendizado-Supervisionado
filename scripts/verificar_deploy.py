"""Verifica um deploy (local ou Render) passando pela API do client, como o navegador faz.

Uso:
    python scripts/verificar_deploy.py                          # http://127.0.0.1:8080
    python scripts/verificar_deploy.py https://regressao-client.onrender.com

ATENÇÃO: termina com o modelo resetado (apaga o treino atual do servidor).
"""
import base64
import io
import os
import sys
import time
import zipfile

import requests

BASE = (sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8080").rstrip("/")
AMOSTRAS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "sample_data")
CHAVE = "7"
resultados = []


def checar(nome, ok, detalhe=""):
    resultados.append(bool(ok))
    print(f"[{'OK ' if ok else 'ERRO'}] {nome}" + (f"  ->  {detalhe}" if detalhe else ""))


def csv_bytes(nome):
    with open(os.path.join(AMOSTRAS, nome), "rb") as f:
        return f.read()


def enviar(rota, campo, conteudo, nome="base.csv"):
    r = requests.post(f"{BASE}{rota}", files={campo: (nome, io.BytesIO(conteudo))}, data={"chave": CHAVE}, timeout=330)
    return r.status_code, r.json()


def sem_coluna_time(conteudo):
    linhas = conteudo.decode().splitlines()
    return ("\n".join(",".join(l.split(",")[:-1]) for l in linhas)).encode()


def com_time_vazio(conteudo):
    linhas = conteudo.decode().splitlines()
    return ("\n".join([linhas[0]] + [",".join(l.split(",")[:-1]) + "," for l in linhas[1:]])).encode()


print(f"Verificando {BASE} (o 1º acesso pode levar ~1 min se o Render estiver dormindo)...")
t0 = time.time()
try:
    r = requests.get(f"{BASE}/api/status", timeout=330)
    checar("Servidor responde", r.status_code == 200, f"{time.time() - t0:.1f}s")
except requests.RequestException as e:
    sys.exit(f"[ERRO] Não consegui acessar {BASE}: {e}")

requests.post(f"{BASE}/api/reset", timeout=60)
st, j = enviar("/api/testar", "test", csv_bytes("PI_test.csv"))
checar("Teste antes do treino é recusado", st == 400, j.get("error", "")[:60])

st, j = enviar("/api/treinar", "train", csv_bytes("PI_train.csv"), "PI_train.csv")
ok = st == 200 and j.get("treinado") and j.get("grafico_treino_png") and j.get("parametros")
checar("Treino", ok, f"R2 esperado={j.get('expectativa_desempenho', {}).get('R2_esperado', float('nan')):.4f}" if ok else str(j))

st, j = enviar("/api/testar", "test", csv_bytes("PI_test.csv"), "PI_test.csv")
if st == 200 and j["avaliacao"]["possui_rotulos"]:
    zf = zipfile.ZipFile(io.BytesIO(base64.b64decode(j["resultado_zip"])))
    checar("Teste COM rótulos (métricas + zip + gráfico)", bool(zf.namelist()) and bool(j["grafico_png"]),
           f"R2 real={j['avaliacao']['desempenho_real']['R2_real']:.4f}")
else:
    checar("Teste COM rótulos", False, str(j))

for nome, f in (("Teste SEM a coluna time", sem_coluna_time), ("Teste com coluna time VAZIA", com_time_vazio)):
    st, j = enviar("/api/testar", "test", f(csv_bytes("PI_test.csv")))
    checar(nome + " (só previsões)", st == 200 and not j["avaliacao"]["possui_rotulos"] and bool(j["resultado_zip"]),
           j["avaliacao"].get("mensagem", "") if st == 200 else str(j))

r = requests.post(f"{BASE}/api/reset", timeout=60)
s = requests.get(f"{BASE}/api/status", timeout=60).json()
checar("Reset", r.status_code == 200 and s["treinado"] is False)

print(f"\n{sum(resultados)}/{len(resultados)} verificações OK")
sys.exit(0 if all(resultados) else 1)

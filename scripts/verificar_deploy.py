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


def como_json(r):
    """Devolve o JSON da resposta; se não for JSON (ex.: página de erro do Render), explica o que veio."""
    try:
        return r.json()
    except ValueError:
        trecho = " ".join(r.text.split())[:160]
        return {"error": f"resposta não-JSON (HTTP {r.status_code}): {trecho!r}"}


def enviar(rota, campo, conteudo, nome="base.csv"):
    try:
        r = requests.post(f"{BASE}{rota}", files={campo: (nome, io.BytesIO(conteudo))}, data={"chave": CHAVE}, timeout=330)
    except requests.RequestException as e:
        return 0, {"error": f"falha de conexão: {type(e).__name__}"}
    return r.status_code, como_json(r)


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
except requests.RequestException as e:
    sys.exit(f"[ERRO] Não consegui acessar {BASE}: {e}")
corpo = como_json(r)
checar("Client e server respondem (/api/status)", r.status_code == 200,
       f"HTTP {r.status_code} em {time.time() - t0:.1f}s" + ("" if r.status_code == 200 else f" | {corpo.get('error', corpo)}"))
if r.status_code != 200:
    sys.exit("\nParei aqui: o client não conseguiu falar com o server. Veja a mensagem acima e a aba Logs do regressao-client.")

requests.post(f"{BASE}/api/reset", timeout=330)
st, j = enviar("/api/testar", "test", csv_bytes("PI_test.csv"))
checar("Teste antes do treino é recusado", st == 400, j.get("error", "")[:60])

st, j = enviar("/api/treinar", "train", csv_bytes("PI_train.csv"), "PI_train.csv")
ok = st == 200 and j.get("treinado") and j.get("grafico_treino_png") and j.get("parametros")
checar("Treino", ok, f"R2 esperado={j.get('expectativa_desempenho', {}).get('R2_esperado', float('nan')):.4f}" if ok else str(j))

st, j = enviar("/api/testar", "test", csv_bytes("PI_test.csv"), "PI_test.csv")
if st == 200 and j["avaliacao"]["possui_rotulos"]:
    zf = zipfile.ZipFile(io.BytesIO(base64.b64decode(j["resultado_zip"])))
    checar("Teste COM rótulos (métricas + CSV + zip + gráfico)",
           bool(zf.namelist()) and bool(j["grafico_png"]) and bool(j["resultado_csv"]),
           f"R2 real={j['avaliacao']['desempenho_real']['R2_real']:.4f}")
else:
    checar("Teste COM rótulos", False, str(j))

def conferir_sem_rotulo(st, j, linhas_esperadas, sem_coluna_time):
    """Sem rótulo: só o CSV de previsões (sem desempenho e sem gráfico)."""
    if st != 200:
        return False, str(j.get("error", j))
    av = j["avaliacao"]
    linhas = base64.b64decode(j["resultado_csv"]).decode().strip().splitlines()
    cabecalho = linhas[0].split(",")
    ok = (
        av["possui_rotulos"] is False
        and "desempenho_real" not in av
        and "grafico_png" not in j
        and "time_previsto" in cabecalho
        and (not sem_coluna_time or "time" not in cabecalho)   # se o arquivo não tinha time, a saída também não
        and len(linhas) - 1 == linhas_esperadas
        and bool(j["resultado_zip"])
    )
    return ok, f"{len(linhas) - 1} previsões, sem métricas e sem gráfico" if ok else f"cabeçalho={cabecalho}, chaves={sorted(j)}"


n_linhas = len(csv_bytes("PI_test.csv").decode().strip().splitlines()) - 1
casos = (
    ("Teste SEM rótulo (arquivo só com time-5..time-1)", csv_bytes("PI_test_sem_rotulo.csv"), True),
    ("Teste SEM rótulo (coluna time vazia)", com_time_vazio(csv_bytes("PI_test.csv")), False),
)
for nome, conteudo, sem_time in casos:
    st, j = enviar("/api/testar", "test", conteudo)
    ok, detalhe = conferir_sem_rotulo(st, j, n_linhas, sem_time)
    checar(nome + " -> só o CSV", ok, detalhe)

r = requests.post(f"{BASE}/api/reset", timeout=330)
s = como_json(requests.get(f"{BASE}/api/status", timeout=330))
checar("Reset", r.status_code == 200 and s.get("treinado") is False)

print(f"\n{sum(resultados)}/{len(resultados)} verificações OK")
sys.exit(0 if all(resultados) else 1)

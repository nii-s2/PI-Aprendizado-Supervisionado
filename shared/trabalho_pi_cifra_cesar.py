"""Módulo convertido de shared/Trabalho__PI_Cifra_Cesar.ipynb."""
import base64
import csv
import io
import json
import math
import random
import urllib.request
import zlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

# === Célula 0 ===
# [notebook] !import base64
# [notebook] !import csv
# [notebook] !import io
# [notebook] !import math
# [notebook] !import random
# [notebook] !import zlib

FEATURE_COLS = ["time-5", "time-4", "time-3", "time-2", "time-1"]
TARGET_COL = "time"

# === Célula 1 ===
# --- 1. TRATAMENTO PONTA-A-PONTA (ZLIB + CIFRA DE CÉSAR + BASE64) ---
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

# === Célula 2 ===
# --- 2. LEITURA E TRATAMENTO DE DADOS (MIN-MAX ) ---
def ler_csv_de_bytes(csv_bytes: bytes, exigir_alvo: bool = True):
    texto = csv_bytes.decode("utf-8")
    leitor = csv.DictReader(io.StringIO(texto))
    fieldnames = leitor.fieldnames or []

    for col in FEATURE_COLS:
        if col not in fieldnames:
            raise ValueError(f"Coluna de entrada ausente: {col}")

    possui_alvo = TARGET_COL in fieldnames
    if exigir_alvo and not possui_alvo:
        raise ValueError(f"Coluna alvo '{TARGET_COL}' ausente no arquivo.")

    X = []
    y = []
    linhas = []

    for linha in leitor:
        if any(linha.get(c) is None or str(linha.get(c)).strip() == "" for c in FEATURE_COLS):
            continue
        vals_x = [float(linha[c]) for c in FEATURE_COLS]

        if exigir_alvo:
            if linha.get(TARGET_COL) is None or str(linha.get(TARGET_COL)).strip() == "":
                continue
            y.append(float(linha[TARGET_COL]))

        X.append(vals_x)
        linhas.append(dict(linha))

    return X, y, linhas, possui_alvo


def ajustar_minmax_matriz(matriz):
    n_cols = len(matriz[0])
    mins, maxs = [], []
    for j in range(n_cols):
        col = [linha[j] for linha in matriz]
        mins.append(min(col))
        maxs.append(max(col))
    return mins, maxs


def aplicar_minmax_matriz(matriz, mins, maxs):
    out = []
    for linha in matriz:
        nova = []
        for j, val in enumerate(linha):
            amp = maxs[j] - mins[j]
            nova.append(0.0 if amp == 0 else (val - mins[j]) / amp)
        out.append(nova)
    return out


def normalizar_vetor(vetor):
    mn = min(vetor)
    mx = max(vetor)
    amp = mx - mn
    if amp == 0:
        return [0.0 for _ in vetor], mn, mx
    return [(v - mn) / amp for v in vetor], mn, mx


def desnormalizar_vetor(vetor_norm, mn, mx):
    amp = mx - mn
    return [mn if amp == 0 else v * amp + mn for v in vetor_norm]

# === Célula 3 ===
# --- 3. REGRESSÃO LINEAR MÚLTIPLA E MÉTRICAS  ---
def treinar_rl(X, y, lr=0.1, epocas=2500):
    n = len(X)
    p = len(X[0])
    pesos = [0.0] * p
    bias = 0.0

    for _ in range(epocas):
        grad_w = [0.0] * p
        grad_b = 0.0
        for i in range(n):
            pred = bias + sum(pesos[j] * X[i][j] for j in range(p))
            erro = pred - y[i]
            grad_b += erro
            for j in range(p):
                grad_w[j] += erro * X[i][j]
        bias -= lr * (grad_b / n)
        for j in range(p):
            pesos[j] -= lr * (grad_w[j] / n)

    return pesos, bias


def prever_rl(X, pesos, bias):
    return [bias + sum(pesos[j] * linha[j] for j in range(len(pesos))) for linha in X]


def calc_mse(y_real, y_pred):
    return sum((r - p) ** 2 for r, p in zip(y_real, y_pred)) / len(y_real)


def calc_mae(y_real, y_pred):
    return sum(abs(r - p) for r, p in zip(y_real, y_pred)) / len(y_real)


def calc_r2(y_real, y_pred):
    media = sum(y_real) / len(y_real)
    ss_res = sum((r - p) ** 2 for r, p in zip(y_real, y_pred))
    ss_tot = sum((r - media) ** 2 for r in y_real)
    return 0.0 if ss_tot == 0 else 1.0 - (ss_res / ss_tot)


def expectativa_k_fold(X, y, k=5, seed=42):
    indices = list(range(len(X)))
    rng = random.Random(seed)
    rng.shuffle(indices)

    k_efetivo = min(k, len(indices))
    tamanho = len(indices) // k_efetivo
    mses, maes, r2s = [], [], []

    for f in range(k_efetivo):
        ini = f * tamanho
        fim = len(indices) if f == k_efetivo - 1 else (f + 1) * tamanho
        idx_val = indices[ini:fim]
        idx_tr = indices[:ini] + indices[fim:]

        X_tr = [X[i] for i in idx_tr]
        y_tr = [y[i] for i in idx_tr]
        X_va = [X[i] for i in idx_val]
        y_va = [y[i] for i in idx_val]

        mins_x, maxs_x = ajustar_minmax_matriz(X_tr)
        X_tr_n = aplicar_minmax_matriz(X_tr, mins_x, maxs_x)
        X_va_n = aplicar_minmax_matriz(X_va, mins_x, maxs_x)
        y_tr_n, min_y, max_y = normalizar_vetor(y_tr)

        w, b = treinar_rl(X_tr_n, y_tr_n)
        pred_va = desnormalizar_vetor(prever_rl(X_va_n, w, b), min_y, max_y)

        mses.append(calc_mse(y_va, pred_va))
        maes.append(calc_mae(y_va, pred_va))
        r2s.append(calc_r2(y_va, pred_va))

    mse_medio = sum(mses) / k_efetivo
    return {
        "MSE_esperado": mse_medio,
        "RMSE_esperado": math.sqrt(mse_medio),
        "MAE_esperado": sum(maes) / k_efetivo,
        "R2_esperado": sum(r2s) / k_efetivo,
    }

# === Célula 4 ===
# --- 4. GERENCIADOR DE ESTADO DO MODELO (TREINO, TESTE E RESET) ---
class ServicoRegressaoPI:
    def __init__(self):
        self.resetar()

    def resetar(self):
        self.pesos = None
        self.bias = None
        self.mins_x = None
        self.maxs_x = None
        self.min_y = None
        self.max_y = None
        self.expectativa = None

    def treinar_pacote(self, pacote_b64: str, deslocamento: int = 7):
        csv_bytes = desempacotar_cesar(pacote_b64, deslocamento)
        X, y, _, _ = ler_csv_de_bytes(csv_bytes, exigir_alvo=True)
        if len(X) < 2:
            raise ValueError("Base de treino precisa de pelo menos 2 linhas válidas.")

        self.expectativa = expectativa_k_fold(X, y, k=5)
        self.mins_x, self.maxs_x = ajustar_minmax_matriz(X)
        X_norm = aplicar_minmax_matriz(X, self.mins_x, self.maxs_x)
        y_norm, self.min_y, self.max_y = normalizar_vetor(y)
        self.pesos, self.bias = treinar_rl(X_norm, y_norm)

        return {
            "status": "treinado",
            "amostras_treino": len(X),
            "expectativa_desempenho": self.expectativa,
        }

    def testar_pacote(self, pacote_b64: str, deslocamento: int = 7):
        if self.pesos is None:
            raise RuntimeError("Modelo não treinado. Realize o treino antes do teste.")

        csv_bytes = desempacotar_cesar(pacote_b64, deslocamento)
        # Primeiro lê sem exigir alvo para descobrir se a coluna 'time' existe e tem valores
        X, _, linhas, possui_col_alvo = ler_csv_de_bytes(csv_bytes, exigir_alvo=False)
        X_norm = aplicar_minmax_matriz(X, self.mins_x, self.maxs_x)
        pred_norm = prever_rl(X_norm, self.pesos, self.bias)
        previsoes = desnormalizar_vetor(pred_norm, self.min_y, self.max_y)

        y_reais = []
        preds_validos = []

        for linha, p in zip(linhas, previsoes):
            linha["time_previsto"] = f"{p:.4f}"
            if possui_col_alvo:
                bruto_y = str(linha.get(TARGET_COL, "")).strip()
                if bruto_y != "":
                    y_reais.append(float(bruto_y))
                    preds_validos.append(p)

        possui_rotulos = possui_col_alvo and len(y_reais) > 0
        avaliacao = {"possui_rotulos": possui_rotulos}

        if possui_rotulos:
            mse_real = calc_mse(y_reais, preds_validos)
            mae_real = calc_mae(y_reais, preds_validos)
            r2_real = calc_r2(y_reais, preds_validos)
            esp = self.expectativa or {}
            avaliacao["desempenho_real"] = {
                "MSE_real": mse_real,
                "RMSE_real": math.sqrt(mse_real),
                "MAE_real": mae_real,
                "R2_real": r2_real,
            }
            avaliacao["desempenho_esperado"] = esp
            avaliacao["comparacao"] = {
                "diff_MSE": mse_real - esp.get("MSE_esperado", 0.0),
                "diff_MAE": mae_real - esp.get("MAE_esperado", 0.0),
                "diff_R2": r2_real - esp.get("R2_esperado", 0.0),
            }
        else:
            avaliacao["mensagem"] = "Teste sem rótulos: previsões geradas sem avaliação."

        # Monta CSV de saída em memória
        buf_csv = io.StringIO()
        colunas_saida = list(linhas[0].keys()) if linhas else FEATURE_COLS + ["time_previsto"]
        escritor = csv.DictWriter(buf_csv, fieldnames=colunas_saida)
        escritor.writeheader()
        escritor.writerows(linhas)

        pacote_resposta = empacotar_cesar(buf_csv.getvalue().encode("utf-8"), deslocamento)
        return {
            "avaliacao": avaliacao,
            "pacote_resposta_b64": pacote_resposta,
        }

# === Célula 6 ===
# [notebook] !import json
# [notebook] !from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
# [notebook] !from nucleo_pi import ServicoRegressaoPI

SERVICO = ServicoRegressaoPI()

# === Célula 7 ===
# [notebook] !python servidor_pi.py
# [notebook] !pip install pyinstaller

# === Célula 8 ===
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

# Note: Garanta também que o módulo nucleo_pi esteja carregado ou definido antes de usar o SERVICO

class HandlerPI(BaseHTTPRequestHandler):
    def _enviar_json(self, codigo_status: int, payload: dict):
        corpo = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(codigo_status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(corpo)))
        self.end_headers()
        self.wfile.write(corpo)

    def _ler_json(self) -> dict:
        tamanho = int(self.headers.get("Content-Length", 0))
        bruto = self.rfile.read(tamanho).decode("utf-8")
        return json.loads(bruto) if bruto else {}

    def do_POST(self):
        try:
            if self.path == "/reset":
                SERVICO.resetar()
                self._enviar_json(200, {"status": "resetado", "mensagem": "Modelo resetado com sucesso."})

            elif self.path == "/treinar":
                dados = self._ler_json()
                deslocamento = int(dados.get("deslocamento", 7))
                resp = SERVICO.treinar_pacote(dados["pacote_b64"], deslocamento=deslocamento)
                self._enviar_json(200, resp)

            elif self.path == "/testar":
                dados = self._ler_json()
                deslocamento = int(dados.get("deslocamento", 7))
                resp = SERVICO.testar_pacote(dados["pacote_b64"], deslocamento=deslocamento)
                self._enviar_json(200, resp)

            else:
                self._enviar_json(404, {"erro": "Rota não encontrada."})
        except Exception as exc:
            self._enviar_json(400, {"erro": str(exc)})


def iniciar_servidor(host="0.0.0.0", porta=8000):
    servidor = ThreadingHTTPServer((host, porta), HandlerPI)
    print(f"Servidor PI rodando em http://{host}:{porta} ...")
    servidor.serve_forever()


if __name__ == "__main__":
    iniciar_servidor()

# === Célula 10 ===
# [notebook] !import json
# [notebook] !import tkinter as tk
# [notebook] !from tkinter import filedialog, messagebox, scrolledtext
# [notebook] !import urllib.request
# [notebook] !from nucleo_pi import desempacotar_cesar, empacotar_cesar

# === Célula 11 ===
def requisicao_post(url: str, payload: dict, timeout: int = 60) -> dict:
    corpo = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=corpo,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


class AppClientePI:
    def __init__(self, root):
        self.root = root
        self.root.title("PI - Regressão Linear Remota (Na Unha + Cifra de César)")
        self.root.geometry("720x500")

        topo = tk.Frame(root)
        topo.pack(fill="x", padx=10, pady=8)

        tk.Label(topo, text="URL do Servidor:").grid(row=0, column=0, sticky="w")
        self.entry_url = tk.Entry(topo, width=42)
        self.entry_url.insert(0, "http://127.0.0.1:8000")
        self.entry_url.grid(row=0, column=1, padx=6)

        tk.Label(topo, text="Chave César:").grid(row=0, column=2, sticky="e")
        self.entry_chave = tk.Entry(topo, width=8)
        self.entry_chave.insert(0, "7")
        self.entry_chave.grid(row=0, column=3, padx=6)

        botoes = tk.Frame(root)
        botoes.pack(pady=6)

        tk.Button(botoes, text="1. Resetar Modelo", command=self.resetar).grid(row=0, column=0, padx=6)
        tk.Button(botoes, text="2. Upload CSV Treino", command=self.treinar).grid(row=0, column=1, padx=6)
        tk.Button(botoes, text="3. Upload CSV Teste", command=self.testar).grid(row=0, column=2, padx=6)

        self.log_area = scrolledtext.ScrolledText(root, width=84, height=22)
        self.log_area.pack(padx=10, pady=10)

    def log(self, texto: str):
        self.log_area.insert(tk.END, texto + "\n")
        self.log_area.see(tk.END)

    def _url_base(self) -> str:
        return self.entry_url.get().strip().rstrip("/")

    def _chave(self) -> int:
        return int(self.entry_chave.get().strip())

    def resetar(self):
        try:
            res = requisicao_post(f"{self._url_base()}/reset", {})
            self.log(f"[RESET] {json.dumps(res, ensure_ascii=False)}")
        except Exception as e:
            messagebox.showerror("Erro no reset", str(e))

    def treinar(self):
        caminho = filedialog.askopenfilename(filetypes=[("Arquivos CSV", "*.csv")])
        if not caminho:
            return
        try:
            with open(caminho, "rb") as f:
                csv_bytes = f.read()
            chave = self._chave()
            pacote = empacotar_cesar(csv_bytes, deslocamento=chave)
            res = requisicao_post(
                f"{self._url_base()}/treinar",
                {"pacote_b64": pacote, "deslocamento": chave},
            )
            self.log(f"[TREINO] {json.dumps(res, ensure_ascii=False, indent=2)}")
        except Exception as e:
            messagebox.showerror("Erro no treino", str(e))

    def testar(self):
        caminho = filedialog.askopenfilename(filetypes=[("Arquivos CSV", "*.csv")])
        if not caminho:
            return
        try:
            with open(caminho, "rb") as f:
                csv_bytes = f.read()
            chave = self._chave()
            pacote = empacotar_cesar(csv_bytes, deslocamento=chave)
            res = requisicao_post(
                f"{self._url_base()}/testar",
                {"pacote_b64": pacote, "deslocamento": chave},
            )
            self.log(f"[AVALIAÇÃO] {json.dumps(res['avaliacao'], ensure_ascii=False, indent=2)}")

            csv_resultado = desempacotar_cesar(res["pacote_resposta_b64"], deslocamento=chave)
            destino = filedialog.asksaveasfilename(
                defaultextension=".csv",
                initialfile="previsoes_time.csv",
                filetypes=[("Arquivos CSV", "*.csv")],
            )
            if destino:
                with open(destino, "wb") as f:
                    f.write(csv_resultado)
                self.log(f"[DOWNLOAD] Arquivo salvo em: {destino}")
        except Exception as e:
            messagebox.showerror("Erro no teste", str(e))


if __name__ == "__main__":
    root = tk.Tk()
    app = AppClientePI(root)
    root.mainloop()

# === Célula 12 ===
# [notebook] !pyinstaller --onefile --add-data "nucleo_pi.py;." cliente_pi.py

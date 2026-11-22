import io
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit


# Função para o modelo: b0 + b1*x1 + ... + b5*x5
def f(x, *b):
    return b[0] + x @ np.array(b[1:])


def metricas(y, y_fit):
    SSE = float(np.sum((y - y_fit) ** 2))
    MSE = SSE / len(y)
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    return {
        "SSE": SSE,
        "MSE": MSE,
        "RMSE": MSE ** 0.5,
        "MAE": float(np.mean(np.abs(y - y_fit))),
        "R2": 0.0 if ss_tot == 0 else 1.0 - (SSE / ss_tot),
    }


def _carregar_csv_input(arquivo_csv):
    """Aceita bytes, str (CSV bruto ou caminho), file-like/upload ou DataFrame."""
    if isinstance(arquivo_csv, pd.DataFrame):
        return arquivo_csv.copy()
    if isinstance(arquivo_csv, bytes):
        return pd.read_csv(io.BytesIO(arquivo_csv))
    if isinstance(arquivo_csv, str):
        if "\n" in arquivo_csv or "," in arquivo_csv or ";" in arquivo_csv:
            return pd.read_csv(io.StringIO(arquivo_csv))
        return pd.read_csv(arquivo_csv)
    if hasattr(arquivo_csv, "read"):
        conteudo = arquivo_csv.read()
        if isinstance(conteudo, bytes):
            return pd.read_csv(io.BytesIO(conteudo))
        return pd.read_csv(io.StringIO(conteudo))
    raise TypeError("Formato de entrada não suportado para o arquivo CSV.")


def treinar(arquivo_treino, alvo="time"):
    """Treina o modelo usando o arquivo/conteúdo enviado pelo input de treino."""
    df = _carregar_csv_input(arquivo_treino)
    colunas_features = [c for c in df.columns if c != alvo]
    X = df[colunas_features].to_numpy(dtype=float)
    y = df[alvo].to_numpy(dtype=float)

    # Normalização min-max (guardada para reaplicar no teste)
    xmin, xmax = X.min(axis=0), X.max(axis=0)
    denom = np.where((xmax - xmin) == 0, 1.0, xmax - xmin)
    Xn = (X - xmin) / denom

    # Chute inicial
    guess = [1.0] * (X.shape[1] + 1)

    # Expectativa de desempenho: hold-out nos últimos 20% (série temporal, sem embaralhar)
    corte = int(len(y) * 0.8)
    p_val, _ = curve_fit(f, Xn[:corte], y[:corte], guess)
    esperado = metricas(y[corte:], f(Xn[corte:], *p_val))

    # Fitting do modelo final (base toda)
    params, params_covariance = curve_fit(f, Xn, y, guess)

    # Métricas no treino
    y_fit = f(Xn, *params)
    treino = metricas(y, y_fit)

    # Plot em alta resolução
    fig, ax = plt.subplots(figsize=(12, 4.5), dpi=300)
    ax.plot(y_fit, color="red", linewidth=1.8, label="Linear Fitting")
    ax.plot(y, color="blue", linewidth=1.8, label="Time Serie")
    ax.legend(loc="best")
    fig.tight_layout()

    return {
        "params": params,
        "xmin": xmin,
        "xmax": xmax,
        "colunas": colunas_features,
        "treino": treino,
        "esperado": esperado,
        "fig": fig,
    }


def prever(modelo, arquivo_teste, alvo="time"):
    """Aplica o modelo no CSV enviado pelo input de teste e retorna DataFrame, métricas e CSV em bytes."""
    df_teste = _carregar_csv_input(arquivo_teste)
    X = df_teste[modelo["colunas"]].to_numpy(dtype=float)
    denom = np.where((modelo["xmax"] - modelo["xmin"]) == 0, 1.0, modelo["xmax"] - modelo["xmin"])
    Xn = (X - modelo["xmin"]) / denom  # mesma normalização do treino

    saida = df_teste.copy()
    saida["time_previsto"] = f(Xn, *modelo["params"])
    met = (
        metricas(saida[alvo].to_numpy(dtype=float), saida["time_previsto"].to_numpy(dtype=float))
        if alvo in saida.columns
        else None
    )
    csv_bytes = saida.to_csv(index=False).encode("utf-8")
    return saida, met, csv_bytes
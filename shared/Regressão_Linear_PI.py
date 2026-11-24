"""Módulo de regressão linear adaptado de Regressão_Linear_PI.ipynb para a v4."""
import io
import os
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

LAGS = 5
TARGET = "time"
FEATURE_COLS = [f"time-{i}" for i in range(LAGS, 0, -1)]  # time-5..time-1


# ---------- modelo linear: b0 + b1*x1 + ... + b5*x5 ----------
def f(x, *b):
    x = np.asarray(x, dtype=float)
    b_arr = np.asarray(b, dtype=float)
    return b_arr[0] + x @ b_arr[1:]


def metricas(y, y_fit):
    y = np.asarray(y, dtype=float)
    y_fit = np.asarray(y_fit, dtype=float)
    if len(y) == 0:
        raise ValueError("Não é possível calcular métricas com vetor vazio.")
    SSE = float(np.sum((y - y_fit) ** 2))
    MSE = float(SSE / len(y))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    return {
        "SSE": SSE,
        "MSE": MSE,
        "RMSE": float(MSE ** 0.5),
        "MAE": float(np.mean(np.abs(y - y_fit))),
        "R2": 0.0 if ss_tot == 0 else float(1.0 - SSE / ss_tot),
    }


def _carregar_csv_input(arquivo_csv):
    """Aceita bytes, str (CSV bruto ou caminho), DataFrame ou objeto file-like."""
    if isinstance(arquivo_csv, pd.DataFrame):
        return arquivo_csv.copy()
    if isinstance(arquivo_csv, bytes):
        try:
            return pd.read_csv(io.BytesIO(arquivo_csv))
        except UnicodeDecodeError as exc:
            raise ValueError("O arquivo não está em UTF-8. Salve o CSV como UTF-8 e tente de novo.") from exc
        except (pd.errors.ParserError, pd.errors.EmptyDataError) as exc:
            raise ValueError("Não foi possível ler o arquivo como CSV (cabeçalho na 1ª linha e vírgula como separador).") from exc
    if isinstance(arquivo_csv, (str, os.PathLike)):
        texto = os.fspath(arquivo_csv)
        if isinstance(texto, str) and ("\n" in texto or "\r" in texto or "," in texto or ";" in texto):
            return pd.read_csv(io.StringIO(texto))
        return pd.read_csv(texto)
    if hasattr(arquivo_csv, "read"):
        conteudo = arquivo_csv.read()
        if isinstance(conteudo, bytes):
            return pd.read_csv(io.BytesIO(conteudo))
        return pd.read_csv(io.StringIO(conteudo))
    raise TypeError("Formato de entrada não suportado para o arquivo CSV.")


def _build_features(df, alvo=TARGET):
    """Retorna (X, y, cols, df_alinhado).
    - Usa time-5..time-1 se existirem.
    - Senão, gera as lags a partir de `alvo` (perde as LAGS primeiras linhas).
    - `y` é None quando `alvo` não está no CSV (teste sem rótulo).
    """
    df = _carregar_csv_input(df)
    if all(c in df.columns for c in FEATURE_COLS):
        try:
            X = df[FEATURE_COLS].to_numpy(dtype=float)
            y = df[alvo].to_numpy(dtype=float) if alvo in df.columns else None
        except (ValueError, TypeError) as exc:
            raise ValueError(f"Há valores não numéricos nas colunas de entrada ou em '{alvo}'.") from exc
        return X, y, list(FEATURE_COLS), df.copy()

    if alvo not in df.columns:
        raise ValueError(
            f"CSV sem colunas de lag e sem coluna '{alvo}' — não há como montar features."
        )

    try:
        s = df[alvo].to_numpy(dtype=float)
    except (ValueError, TypeError) as exc:
        raise ValueError(f"Há valores não numéricos na coluna '{alvo}'.") from exc
    if len(s) <= LAGS:
        raise ValueError(
            f"Série muito curta ({len(s)} linhas) para gerar {LAGS} lags."
        )
    X = np.array([s[i - LAGS:i] for i in range(LAGS, len(s))], dtype=float)
    y = s[LAGS:].astype(float)
    df_alin = df.iloc[LAGS:].copy()
    return X, y, list(FEATURE_COLS), df_alin


# ---------- treino ----------
def treinar(df, alvo=TARGET):
    X, y, cols, _ = _build_features(df, alvo=alvo)
    if y is None:
        raise ValueError("CSV de treino precisa ter a coluna alvo.")
    if len(y) < 8:
        raise ValueError("A base de treino precisa de pelo menos 8 linhas para o ajuste e a validação.")
    if not np.isfinite(X).all() or not np.isfinite(y).all():
        raise ValueError("A base de treino contém valores ausentes ou não numéricos.")

    xmin, xmax = X.min(axis=0), X.max(axis=0)
    denom = np.where((xmax - xmin) == 0, 1.0, xmax - xmin)
    Xn = (X - xmin) / denom

    guess = [1.0] * (X.shape[1] + 1)
    corte = int(len(y) * 0.8)
    p_val, _ = curve_fit(f, Xn[:corte], y[:corte], guess, maxfev=10000)
    esperado = metricas(y[corte:], f(Xn[corte:], *p_val))

    params, params_cov = curve_fit(f, Xn, y, guess, maxfev=10000)
    y_fit = f(Xn, *params)
    treino = metricas(y, y_fit)

    modelo = {
        "params": params,
        "params_cov": params_cov,
        "xmin": xmin,
        "xmax": xmax,
        "denom": denom,
        "colunas": cols,
        "alvo": alvo,
        "treino": treino,
        "esperado": esperado,
    }
    return modelo


# ---------- teste / aplicação ----------
def prever(modelo, df_teste, alvo=TARGET):
    alvo_efetivo = alvo or modelo.get("alvo", TARGET)
    X, y_true, _, df_alin = _build_features(df_teste, alvo=alvo_efetivo)

    if X.shape[1] != len(modelo["colunas"]):
        raise ValueError(
            f"Shape inesperado: teste tem {X.shape[1]} features, "
            f"modelo espera {len(modelo['colunas'])}."
        )
    if not np.isfinite(X).all():
        raise ValueError("A base de teste contém valores ausentes ou não numéricos.")

    Xn = (X - modelo["xmin"]) / modelo["denom"]
    df_alin = df_alin.copy()
    df_alin[f"{alvo_efetivo}_previsto"] = f(Xn, *modelo["params"])

    met = None
    if y_true is not None:
        previsto = df_alin[f"{alvo_efetivo}_previsto"].to_numpy(dtype=float)
        com_rotulo = np.isfinite(y_true)           # linhas com 'time' vazio ficam sem avaliação
        if com_rotulo.any():
            met = metricas(y_true[com_rotulo], previsto[com_rotulo])

    csv_bytes = df_alin.to_csv(index=False).encode("utf-8")
    return df_alin, met, csv_bytes

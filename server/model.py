"""Treino e teste da regressão linear + geração do gráfico."""
import io

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Paleta do projeto
NAVY, GOLD, PEACH, TEAL, BLUE = "#131A3C", "#E6C17A", "#FDD595", "#315D6B", "#224B68"


def _numeric(df: pd.DataFrame, cols: list[str], nome: str) -> pd.DataFrame:
    try:
        return df[cols].apply(pd.to_numeric, errors="raise")
    except ValueError as exc:
        raise ValueError(f"A base de {nome} tem colunas não numéricas: {exc}") from exc


def run_regression(train_df: pd.DataFrame, test_df: pd.DataFrame, target: str | None):
    target = (target or "").strip() or train_df.columns[-1]
    for nome, df in (("treino", train_df), ("teste", test_df)):
        if target not in df.columns:
            raise ValueError(f"A coluna alvo '{target}' não existe na base de {nome}.")

    features = [c for c in train_df.columns if c != target]
    if not features:
        raise ValueError("A base de treino precisa de ao menos uma coluna de entrada.")
    faltando = [c for c in features if c not in test_df.columns]
    if faltando:
        raise ValueError(f"Colunas ausentes na base de teste: {', '.join(faltando)}.")

    train_df = train_df.dropna(subset=features + [target])
    test_df = test_df.dropna(subset=features + [target]).reset_index(drop=True)
    if len(train_df) < 2 or len(test_df) < 1:
        raise ValueError("Linhas insuficientes após remover valores vazios.")

    X_train, y_train = _numeric(train_df, features, "treino"), _numeric(train_df, [target], "treino")[target]
    X_test, y_test = _numeric(test_df, features, "teste"), _numeric(test_df, [target], "teste")[target]

    model = LinearRegression().fit(X_train, y_train)
    pred = model.predict(X_test)

    mse = mean_squared_error(y_test, pred)
    metrics = {
        "target": target,
        "features": features,
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "r2": float(r2_score(y_test, pred)) if len(y_test) > 1 else None,
        "mse": float(mse),
        "rmse": float(mse ** 0.5),
        "mae": float(mean_absolute_error(y_test, pred)),
        "intercept": float(model.intercept_),
        "coefficients": {f: float(c) for f, c in zip(features, model.coef_)},
    }

    results = X_test.copy()
    results[f"{target}_real"] = y_test.values
    results[f"{target}_previsto"] = pred
    results["erro"] = y_test.values - pred

    return metrics, results, _plot(X_test, y_test, pred, features, target)


def _style(ax):
    ax.set_facecolor(NAVY)
    for s in ax.spines.values():
        s.set_color(BLUE)
    ax.tick_params(colors=PEACH)
    ax.xaxis.label.set_color(PEACH)
    ax.yaxis.label.set_color(PEACH)
    ax.title.set_color(GOLD)
    ax.grid(color=BLUE, alpha=0.5, linewidth=0.6)


def _plot(X_test, y_test, pred, features, target) -> bytes:
    single = len(features) == 1
    fig, axes = plt.subplots(1, 2 if single else 1, figsize=(12 if single else 6.5, 5), facecolor=NAVY)
    axes = list(axes) if single else [axes]

    if single:
        ax, f = axes[0], features[0]
        order = X_test[f].argsort()
        ax.scatter(X_test[f], y_test, color=GOLD, s=26, alpha=0.85, label="Real")
        ax.plot(X_test[f].iloc[order], pred[order], color=PEACH, linewidth=2, label="Reta ajustada")
        ax.set_xlabel(f)
        ax.set_ylabel(target)
        ax.set_title("Dados de teste e reta ajustada")
        ax.legend(facecolor=BLUE, edgecolor=BLUE, labelcolor=PEACH)
        _style(ax)

    ax = axes[-1]
    lo, hi = min(y_test.min(), pred.min()), max(y_test.max(), pred.max())
    ax.plot([lo, hi], [lo, hi], color=TEAL, linestyle="--", linewidth=1.5, label="Previsão perfeita")
    ax.scatter(y_test, pred, color=GOLD, s=26, alpha=0.85)
    ax.set_xlabel("Valor real")
    ax.set_ylabel("Valor previsto")
    ax.set_title("Real × previsto (base de teste)")
    ax.legend(facecolor=BLUE, edgecolor=BLUE, labelcolor=PEACH)
    _style(ax)

    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=140, facecolor=NAVY)
    plt.close(fig)
    return buf.getvalue()

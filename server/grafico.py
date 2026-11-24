"""Gráfico do teste: série real x prevista e dispersão real x prevista em alta resolução."""
import io
import math
import os

import matplotlib

matplotlib.use("Agg")   # O servidor só gera PNG
import matplotlib.pyplot as plt

DPI = int(os.environ.get("GRAFICO_DPI", "300"))   # no Render gratuito (0,1 CPU) use 200

NAVY, STEEL, SLATE, GOLD, PEACH, ICE = "#131A3C", "#224B68", "#315D6B", "#E6C17A", "#FDD595", "#7FB0BF"


def _estilo(ax):
    ax.set_facecolor(NAVY)
    for s in ax.spines.values():
        s.set_color(STEEL)
        s.set_linewidth(1.1)
    ax.tick_params(colors=PEACH, labelsize=11)
    ax.xaxis.label.set_color(PEACH)
    ax.xaxis.label.set_fontsize(12)
    ax.yaxis.label.set_color(PEACH)
    ax.yaxis.label.set_fontsize(12)
    ax.title.set_color(GOLD)
    ax.title.set_fontsize(13.5)
    ax.title.set_fontweight("bold")
    ax.grid(color=STEEL, alpha=0.55, linewidth=0.8)


def gerar_grafico(reais: list[float], previsoes: list[float]) -> bytes:
    """`reais` pode ter NaN nas linhas sem rótulo."""
    pares = [(r, p) for r, p in zip(reais, previsoes) if not math.isnan(r)]
    tem_reais = bool(pares)
    fig, axes = plt.subplots(
        1,
        2 if tem_reais else 1,
        figsize=(16 if tem_reais else 9.5, 6.2),
        dpi=DPI,
        facecolor=NAVY,
    )
    axes = list(axes) if tem_reais else [axes]

    ax = axes[0]
    x = list(range(1, len(previsoes) + 1))
    if tem_reais:
        ax.plot(x, reais, color=GOLD, linewidth=2.2, label="Real (time)")
    ax.plot(x, previsoes, color=ICE, linewidth=2.2, linestyle="--", label="Previsto")
    ax.set_xlabel("Linha da base de teste")
    ax.set_ylabel("time")
    ax.set_title("Série real e prevista")
    ax.legend(facecolor=STEEL, edgecolor=STEEL, labelcolor=PEACH, fontsize=11)
    _estilo(ax)

    if tem_reais:
        ax = axes[1]
        rs, ps = [p[0] for p in pares], [p[1] for p in pares]
        lo, hi = min(rs + ps), max(rs + ps)
        ax.plot([lo, hi], [lo, hi], color=SLATE, linestyle="--", linewidth=1.8, label="Previsão perfeita")
        ax.scatter(rs, ps, color=GOLD, s=36, alpha=0.88, edgecolors="none")
        ax.set_xlabel("Valor real")
        ax.set_ylabel("Valor previsto")
        ax.set_title("Real × previsto")
        ax.legend(facecolor=STEEL, edgecolor=STEEL, labelcolor=PEACH, fontsize=11)
        _estilo(ax)

    fig.tight_layout(pad=1.6)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=DPI, facecolor=NAVY, bbox_inches="tight")
    plt.close(fig)
    return buf.getvalue()


def gerar_grafico_treino(reais: list[float], ajuste: list[float]) -> bytes:
    """Mesmo gráfico do notebook: ajuste linear x série temporal na base de treino."""
    fig, ax = plt.subplots(figsize=(13, 5.2), dpi=DPI, facecolor=NAVY)
    x = list(range(len(reais)))
    ax.plot(x, ajuste, color=ICE, linewidth=2.0, label="Linear Fitting")
    ax.plot(x, reais, color=GOLD, linewidth=2.0, label="Time Serie")
    ax.set_xlabel("Linha da base de treino")
    ax.set_ylabel("time")
    ax.set_title("Ajuste linear na base de treino")
    ax.margins(x=0.01)
    ax.legend(facecolor=STEEL, edgecolor=STEEL, labelcolor=PEACH, fontsize=11, loc="best")
    _estilo(ax)

    fig.tight_layout(pad=1.6)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=DPI, facecolor=NAVY, bbox_inches="tight")
    plt.close(fig)
    return buf.getvalue()

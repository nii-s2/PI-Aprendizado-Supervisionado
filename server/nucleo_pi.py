"""Camada de integração entre a API Flask e o modelo definido em shared/Regressão_Linear_PI.py."""
import importlib.util
import math
import os
import sys

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from shared.compress_encrypt import empacotar_cesar
from shared.decompress_decrypt import desempacotar_cesar
from shared.trabalho_pi_cifra_cesar import NOME_API_CIFRA_CESAR
from server.grafico import gerar_grafico, gerar_grafico_treino

NOME_API_REGRESSAO = "Regressão_Linear_PI"
MODELO_PATH = os.path.join(ROOT_DIR, "shared", "Regressão_Linear_PI.py")
_spec = importlib.util.spec_from_file_location("modelo_regressao_linear_pi_v4", MODELO_PATH)
if _spec is None or _spec.loader is None:
    raise ImportError(f"Não foi possível carregar o modelo em: {MODELO_PATH}")
MODELO = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(MODELO)


class ServicoRegressaoPI:
    """Mantém o modelo treinado em memória e adapta Regressão_Linear_PI à API HTTP."""

    def __init__(self, alvo: str = "time"):
        self.alvo = alvo
        self.resetar()

    def resetar(self):
        self.modelo = None
        self.amostras_treino = 0
        self.expectativa = None
        self.metricas_treino = None
        self.parametros = None
        self.amostras_holdout = 0

    def status(self) -> dict:
        return {
            "treinado": self.modelo is not None,
            "api": NOME_API_REGRESSAO,
            "cifra_api": NOME_API_CIFRA_CESAR,
            "amostras_treino": self.amostras_treino if self.modelo is not None else 0,
            "expectativa_desempenho": self.expectativa if self.modelo is not None else None,
            "metricas_treino": self.metricas_treino if self.modelo is not None else None,
            "parametros": self.parametros if self.modelo is not None else None,
            "amostras_holdout": self.amostras_holdout if self.modelo is not None else 0,
        }

    def treinar_pacote(self, pacote_b64: str, deslocamento: int = 7) -> dict:
        csv_bytes = desempacotar_cesar(pacote_b64, deslocamento)
        modelo = MODELO.treinar(csv_bytes, alvo=self.alvo)

        treino = modelo["treino"]
        esperado = modelo["esperado"]
        X, y, cols, _ = MODELO._build_features(csv_bytes, alvo=self.alvo)

        # mesmo ajuste do notebook (entradas normalizadas), usado no gráfico e na exibição dos parâmetros
        y_ajuste = MODELO.f((X - modelo["xmin"]) / modelo["denom"], *modelo["params"])
        nomes = ["b0 (intercepto)"] + [f"b{i} · {c}" for i, c in enumerate(cols, start=1)]
        self.parametros = [
            {"nome": n, "valor": float(v)} for n, v in zip(nomes, modelo["params"])
        ]
        self.amostras_holdout = int(len(y) - int(len(y) * 0.8))   # mesmo corte do hold-out

        self.expectativa = {
            "SSE_esperado": float(esperado["SSE"]),
            "MSE_esperado": float(esperado["MSE"]),
            "RMSE_esperado": float(esperado["RMSE"]),
            "MAE_esperado": float(esperado["MAE"]),
            "R2_esperado": float(esperado["R2"]),
        }
        self.metricas_treino = {k: float(v) for k, v in treino.items()}
        self.modelo = modelo
        self.amostras_treino = int(len(X))

        return {
            "status": "treinado",
            "treinado": True,
            "api": NOME_API_REGRESSAO,
            "cifra_api": NOME_API_CIFRA_CESAR,
            "amostras_treino": self.amostras_treino,
            "expectativa_desempenho": self.expectativa,
            "metricas_treino": self.metricas_treino,
            "parametros": self.parametros,
            "amostras_holdout": self.amostras_holdout,
            "pacote_grafico_treino_b64": empacotar_cesar(
                gerar_grafico_treino(y.tolist(), y_ajuste.tolist()), deslocamento
            ),
        }

    def testar_pacote(self, pacote_b64: str, deslocamento: int = 7) -> dict:
        if self.modelo is None:
            raise RuntimeError("Modelo não treinado. Realize o treino antes do teste. "
                               "(Se o servidor reiniciou ou estava inativo, o modelo em memória foi perdido: treine de novo.)")

        csv_bytes = desempacotar_cesar(pacote_b64, deslocamento)
        saida, met, csv_resultado = MODELO.prever(self.modelo, csv_bytes, alvo=self.alvo)
        col_prev = f"{self.alvo}_previsto"
        previsoes = saida[col_prev].astype(float).tolist()
        reais = (
            saida[self.alvo].astype(float).tolist()
            if self.alvo in saida.columns
            else [float("nan")] * len(saida)
        )

        possui_rotulos = met is not None
        avaliacao = {
            "possui_rotulos": possui_rotulos,
            "amostras_teste": int(len(saida)),
        }
        if possui_rotulos:
            n_rotuladas = int(saida[self.alvo].notna().sum())
            avaliacao["amostras_avaliadas"] = n_rotuladas
            if n_rotuladas < len(saida):
                avaliacao["mensagem"] = (
                    f"Avaliação feita em {n_rotuladas} das {len(saida)} linhas: "
                    f"as demais estão sem '{self.alvo}' e só receberam previsão."
                )
        if possui_rotulos:
            sse = float(met["SSE"])
            mse = float(met["MSE"])
            mae = float(met["MAE"])
            r2 = float(met["R2"])
            avaliacao["desempenho_real"] = {
                "SSE_real": sse,
                "MSE_real": mse,
                "RMSE_real": float(met.get("RMSE", math.sqrt(mse))),
                "MAE_real": mae,
                "R2_real": r2,
            }
            avaliacao["desempenho_esperado"] = self.expectativa
            avaliacao["comparacao"] = {
                "diff_MSE": mse - self.expectativa["MSE_esperado"],
                "diff_MAE": mae - self.expectativa["MAE_esperado"],
                "diff_R2": r2 - self.expectativa["R2_esperado"],
            }
        else:
            avaliacao["mensagem"] = (
                f"Teste sem rótulos (sem a coluna '{self.alvo}' preenchida): foram geradas apenas as "
                "previsões, sem avaliação de desempenho."
            )

        resposta = {
            "api": NOME_API_REGRESSAO,
            "cifra_api": NOME_API_CIFRA_CESAR,
            "avaliacao": avaliacao,
            "pacote_resposta_b64": empacotar_cesar(csv_resultado, deslocamento),
        }
        # O gráfico compara real x previsto: só faz sentido (e só é gerado) quando há rótulos.
        if possui_rotulos:
            resposta["pacote_grafico_b64"] = empacotar_cesar(gerar_grafico(reais, previsoes), deslocamento)
        return resposta

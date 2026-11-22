"""Núcleo do servidor: integra shared/trabalho_pi_cifra_cesar.py com geração de gráfico e status."""
import csv
import io
import math
import os
import sys

# Garante que a raiz do projeto esteja no sys.path para importar `shared`
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from shared.trabalho_pi_cifra_cesar import (
    FEATURE_COLS,
    TARGET_COL,
    ServicoRegressaoPI as ServicoBaseNotebook,
    ajustar_minmax_matriz,
    aplicar_minmax_matriz,
    calc_mae,
    calc_mse,
    calc_r2,
    desempacotar_cesar,
    desnormalizar_vetor,
    empacotar_cesar,
    expectativa_k_fold,
    ler_csv_de_bytes,
    normalizar_vetor,
    prever_rl,
    treinar_rl,
)
from grafico import gerar_grafico


class ServicoRegressaoPI(ServicoBaseNotebook):
    """Estende ServicoRegressaoPI do notebook para expor `/status` e o gráfico criptografado."""

    def __init__(self):
        super().__init__()

    def resetar(self):
        super().resetar()
        self.amostras_treino = 0

    def status(self) -> dict:
        treinado = self.pesos is not None
        return {
            "treinado": treinado,
            "amostras_treino": self.amostras_treino if treinado else 0,
            "expectativa_desempenho": self.expectativa if treinado else None,
        }

    def treinar_pacote(self, pacote_b64: str, deslocamento: int = 7) -> dict:
        resultado = super().treinar_pacote(pacote_b64, deslocamento=deslocamento)
        self.amostras_treino = resultado.get("amostras_treino", 0)
        return {
            "treinado": True,
            **resultado,
        }

    def testar_pacote(self, pacote_b64: str, deslocamento: int = 7) -> dict:
        if self.pesos is None:
            raise RuntimeError("Modelo não treinado. Realize o treino antes do teste.")

        csv_bytes = desempacotar_cesar(pacote_b64, deslocamento)
        X, _, linhas, possui_col_alvo = ler_csv_de_bytes(csv_bytes, exigir_alvo=False)
        if not X:
            raise ValueError("Nenhuma linha válida encontrada na base de teste.")

        X_norm = aplicar_minmax_matriz(X, self.mins_x, self.maxs_x)
        pred_norm = prever_rl(X_norm, self.pesos, self.bias)
        previsoes = desnormalizar_vetor(pred_norm, self.min_y, self.max_y)

        y_reais = []
        preds_validos = []
        reais_plot = []

        for linha, p in zip(linhas, previsoes):
            linha["time_previsto"] = f"{p:.4f}"
            if possui_col_alvo:
                bruto_y = str(linha.get(TARGET_COL, "")).strip()
                if bruto_y != "":
                    val_y = float(bruto_y)
                    y_reais.append(val_y)
                    preds_validos.append(p)
                    reais_plot.append(val_y)
                else:
                    reais_plot.append(float("nan"))
            else:
                reais_plot.append(float("nan"))

        possui_rotulos = possui_col_alvo and len(y_reais) > 0
        avaliacao = {
            "possui_rotulos": possui_rotulos,
            "amostras_teste": len(linhas),
        }

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

        buf_csv = io.StringIO()
        colunas_saida = list(linhas[0].keys()) if linhas else FEATURE_COLS + ["time_previsto"]
        escritor = csv.DictWriter(buf_csv, fieldnames=colunas_saida)
        escritor.writeheader()
        escritor.writerows(linhas)

        grafico_bytes = gerar_grafico(reais_plot, previsoes)

        return {
            "avaliacao": avaliacao,
            "pacote_resposta_b64": empacotar_cesar(buf_csv.getvalue().encode("utf-8"), deslocamento),
            "pacote_grafico_b64": empacotar_cesar(grafico_bytes, deslocamento),
        }

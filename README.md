# Projeto Integrador

Prevê o valor atual de uma **série temporal** (coluna `time`) a partir das **5 observações anteriores** (`time-5` … `time-1`, ou geradas automaticamente se apenas `time` for fornecida). As bases são **compactadas (`zlib`) e criptografadas (cifra de César)** pelo client com a chave informada na interface web, e o server local treina e testa o modelo usando `shared/Regressão_Linear_PI.py`, devolvendo o CSV compactado (`.zip`) e o gráfico (`.png`).

| Arquivo | Função |
|---|---|
| `shared/trabalho_pi_cifra_cesar.py` | Arquivo `.py` convertido de `Trabalho__PI_Cifra_Cesar.ipynb` (compactação `zlib`, cifra de César, leitura CSV, normalização Min-Max, regressão e `ServicoRegressaoPI`) |
| `client/web/` | Interface Web local: `index.html`, `css/style.css`, `js/app.js` |
| `client/app.py` | Servidor local de HTML + API (`/api/status`, `/api/reset`, `/api/treinar`, `/api/testar`) |
| `server/app.py` | Servidor local de processamento (`/status`, `/reset`, `/treinar`, `/testar`) |
| `server/nucleo_pi.py` | Integra `ServicoRegressaoPI` de `shared/trabalho_pi_cifra_cesar.py` com `/status` e geração de gráfico |
| `server/grafico.py` | Gera o gráfico comparativo (`série real × prevista` e dispersão) |
| `rodar_local.py` | Inicia automaticamente o `server` (`127.0.0.1:5000`) e o `client` (`127.0.0.1:8080`) |

## Como rodar localmente

1. Instale as dependências:
```bash
pip install -r requirements.txt
```

2. Execute em um único comando:
```bash
python rodar_local.py
```

2. Ou abra dois terminais separados:
```bash
# Terminal 1 (Server)
python server/app.py

# Terminal 2 (Client Web)
python client/app.py
```

3. Acesse `http://127.0.0.1:8080` no navegador e teste com `sample_data/PI_train.csv` e `sample_data/PI_test.csv` (ou `sample_data/treino.csv` e `sample_data/teste.csv`).

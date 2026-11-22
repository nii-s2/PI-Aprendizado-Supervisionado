# Regressão Segura

Envie uma base de **treino** e uma de **teste** (CSV). Elas são **compactadas (gzip) e criptografadas (AES-256-GCM)** pelo client, descompactadas e descriptografadas pelo server, que treina uma **regressão linear**, testa e devolve o **CSV de resultados compactado** e o **gráfico**.

## Arquitetura

```
Navegador ──CSV──► client (Flask :8080) ──.enc──► server (Flask :5000, interno)
 HTML/CSS/JS        compress_encrypt.py            decompress_decrypt.py
                    decompress_decrypt.py  ◄─.enc─ compress_encrypt.py
                    (resposta)                     LinearRegression + gráfico
```

| Pasta | Função |
|---|---|
| `client/web/` | Site: `index.html`, `css/style.css`, `js/app.js` |
| `client/app.py` | API do client: recebe os CSVs, compacta + criptografa, chama o server, descriptografa a resposta |
| `server/app.py` | API do server: `POST /train`, `GET /health` |
| `server/model.py` | Treino, teste, métricas e gráfico (matplotlib) |
| `shared/compress_encrypt.py` | gzip → AES-256-GCM |
| `shared/decompress_decrypt.py` | AES-256-GCM → gunzip |
| `sample_data/` | CSVs de exemplo |

## Como rodar

```bash
cp .env.example .env
# gere a chave e cole em ENCRYPTION_KEY dentro do .env:
python -c "import os,base64;print(base64.b64encode(os.urandom(32)).decode())"

docker compose up --build
```

Abra http://localhost:8080 e envie `sample_data/treino.csv` e `sample_data/teste.csv`.

## Formato dos CSVs

- Primeira linha com os nomes das colunas.
- Todas as colunas numéricas.
- A coluna alvo é a **última**, ou a informada no campo "Coluna alvo".
- Treino e teste com as mesmas colunas (o teste precisa do alvo para calcular o erro).

## Endpoints

| Método | Rota | Onde | Descrição |
|---|---|---|---|
| POST | `/api/process` | client | `multipart`: `train`, `test`, `target` (opcional). Devolve métricas, `results_csv_gz` e `plot_png` em base64 |
| POST | `/train` | server | `multipart` com `train` e `test` já criptografados. Devolve métricas + resultados criptografados |
| GET | `/health` | server | Verificação de saúde |

## Segurança

- AES-256-GCM com nonce aleatório por arquivo; o GCM também detecta adulteração.
- A chave fica só em `.env` (ignorado pelo git) e é compartilhada por client e server.
- O server não é publicado no host, só responde na rede interna do compose.
- Entre navegador e client o tráfego é local (HTTP). Fora do ambiente de desenvolvimento, coloque HTTPS (proxy reverso) na frente do client.

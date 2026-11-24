# Projeto Integrador — v6 (Nuvem no Render + execução local)

Prevê o valor atual de uma **série temporal** (coluna `time`) a partir das **5 observações anteriores** (`time-5` … `time-1`, ou geradas automaticamente se apenas `time` for fornecida). As bases são **compactadas (`zlib`) e criptografadas (cifra de César)** pelo client com a chave informada na interface web, e o server local treina e testa o modelo usando `shared/Regressão_Linear_PI.py`, devolvendo o CSV compactado (`.zip`) e o gráfico (`.png`).

## Arquitetura (nuvem)

```
Navegador ──► regressao-client (Render)  ──pacote zlib+César──►  regressao-server (Render)
              site HTML + API                                     treina e testa o modelo
              compacta e criptografa                              (Regressão_Linear_PI.py)
              ◄── CSV (.zip) + gráficos ◄── resposta cifrada ◄──
```

* **regressao-server:** treina e executa o modelo em nuvem (`/treinar`, `/testar`, `/reset`, `/status`, `/health`).
* **regressao-client:** serve o site e chama o server; é o endereço que você abre e apresenta.
* O server exige o cabeçalho `X-API-Token` (valor gerado pelo Render e copiado para o client), para que ninguém além do client consiga treinar, testar ou resetar o modelo. O `/health` fica aberto.

## Deploy no Render

1. Coloque o conteúdo desta pasta na **raiz** de um repositório no GitHub (o `render.yaml` precisa estar na raiz):
   ```bash
   git init && git add . && git commit -m "v6" && git branch -M main
   git remote add origin <URL-DO-SEU-REPOSITORIO> && git push -u origin main
   ```
2. No painel do Render: **New + → Blueprint**, escolha o repositório e confirme. Serão criados `regressao-server` e `regressao-client` (plano Free). O primeiro build demora alguns minutos (pandas, scipy, matplotlib).
3. Abra o serviço `regressao-server` e veja sua URL. Se **não** for exatamente `https://regressao-server.onrender.com` (o Render pode acrescentar um sufixo), edite a variável `SERVER_URL` do `regressao-client` (*Environment*) e salve.
4. Acesse a URL do `regressao-client` e verifique o deploy:
   ```bash
   pip install requests
   python scripts/verificar_deploy.py https://regressao-client.onrender.com
   ```
   O script treina com `PI_train.csv`, testa com e sem rótulos (inclusive com `time` vazio) e termina resetando o modelo.

Se o Render não aceitar o Blueprint no seu plano, crie os dois **Web Services** manualmente (runtime Python, plano Free) com os mesmos *Build Command*, *Start Command* e variáveis de ambiente do `render.yaml`. A variável `API_TOKEN` do client deve ter o mesmo valor da do server.

### Limites do plano gratuito (importante para a apresentação)

* **Dorme após 15 min sem acesso** e leva cerca de 1 minuto para acordar. O client já acorda o server em paralelo e espera até ~4 min; a tela mostra "Acordando o servidor…".
* **O modelo fica em memória**: ao dormir, reiniciar ou fazer novo deploy, o servidor volta como *Não treinado*. **Treine pouco antes de apresentar.** Se o teste acusar "Modelo não treinado", basta treinar de novo.
* Há **um único modelo compartilhado** por todos que usam o mesmo server.
* 512 MB de RAM e 0,1 CPU: por isso o gráfico usa `GRAFICO_DPI=200` no Render (300 localmente).
* Mantenha o servidor acordado só perto da apresentação (por exemplo, um ping em `/health` a cada 10 min); 24 h por dia gastaria as 750 horas mensais gratuitas.

## Teste com e sem rótulos (para a apresentação)

* **Treino:** o `PI_train.csv` sempre tem a coluna `time`. As colunas `time-5` … `time-1` são as entradas (X) e `time` é o rótulo (y).
* **Teste COM rótulo** (tem `time`): o sistema prevê, compara com o valor real e mostra R², RMSE, MAE, MSE, SSE e a diferença para o esperado, além do gráfico. O CSV de previsões (`time_previsto`) fica disponível em `.csv` e `.zip`.
* **Teste SEM rótulo** (só `time-5` … `time-1`, ou coluna `time` vazia): o sistema **só prevê**. A tela mostra apenas o aviso e o download do CSV (e do `.zip`) de previsões, **sem desempenho e sem gráfico**.
* O sistema verifica se `time` existe (e está preenchida) no teste e escolhe o caso sozinho.

Arquivos de exemplo em `sample_data/`: `PI_train.csv` (treino), `PI_test.csv` (teste com rótulo) e `PI_test_sem_rotulo.csv` (teste sem rótulo, 114 linhas). Sugestão: treine uma vez e rode os dois testes em sequência.

## Novidades da v6

* **Pronto para o Render:** `render.yaml` (Blueprint com os dois serviços), gunicorn com 1 worker no server, porta via `PORT`, `requirements.txt` separados por serviço e Python 3.12 fixado.
* **Token de acesso** entre client e server (`API_TOKEN`), gerado pelo Render.
* **Cold start tratado:** o client acorda o server e tenta de novo até ele responder; o status inicial mostra um aviso.
* **Teste sem rótulos:** devolve apenas o CSV de previsões (sem desempenho e sem gráfico). Coluna `time` vazia também conta como sem rótulo (antes o servidor devolvia `NaN`, que o navegador não consegue ler). Se só parte das linhas estiver sem `time`, a avaliação usa as linhas rotuladas e a tela informa quantas.
* **JSON sem `NaN`:** valores não finitos viram `null` e aparecem como "–".
* **Mensagens de erro em português** para CSV não numérico, binário ou fora do UTF-8.
* `scripts/verificar_deploy.py` para testar o deploy ponta a ponta.

## Novidades da v5

Correções para que a tela mostre os mesmos resultados do notebook `Regressão_Linear_PI.ipynb`:

* **Título do desempenho esperado corrigido:** o método é *hold-out cronológico (últimos 20% das linhas)*, não validação cruzada.
* **Desempenho no treino** (ajuste com a base toda) agora aparece na tela, com R², RMSE, MAE, MSE e SSE, além do número de linhas.
* **SSE** exibido também no desempenho esperado e no desempenho no teste.
* **Parâmetros do modelo** (`b0` e `b1..b5`, sobre as entradas normalizadas por min-max, 2 casas como no notebook) em uma tabela.
* **Gráfico do treino** (*Linear Fitting* × *Time Serie*, igual ao do notebook) exibido e disponível para download. Ele vem na resposta do treino; se a página for recarregada, as métricas e os parâmetros continuam na tela, mas o gráfico some até um novo treino.

Valores de referência com `sample_data/PI_train.csv` e `sample_data/PI_test.csv` (parâmetros: `114.95 -180. -130.62 8.55 182.37 191.49`):

| | R² | MSE | MAE |
|---|---|---|---|
| Esperado (hold-out, 53 linhas) | 0,3824 | 5786,48 | 59,02 |
| Treino (265 linhas) | 0,4385 | 7008,44 | 61,83 |
| Teste (114 linhas) | 0,3107 | 12760,57 | 82,06 |

Os dígitos finais do SSE podem variar na 2ª ou 3ª casa decimal conforme a versão do SciPy (o `curve_fit` é iterativo), sem relação com o site.

## Novidades da v4

* Modelo de regressão linear atualizado a partir de `Regressão_Linear_PI.ipynb` em `shared/Regressão_Linear_PI.py`.
* `ServicoRegressaoPI` adaptado para consumir `Regressão_Linear_PI.py` (normalização Min-Max, hold-out cronológico de 20%, ajuste via `scipy.optimize.curve_fit`, geração automática de lags caso `time-5..time-1` não existam, e suporte a bases de teste com ou sem rótulo `time`).
* Chamadas de API alinhadas com o nome do modelo `Regressão_Linear_PI` e `trabalho_pi_cifra_cesar`.

## Como rodar localmente

1. Instale as dependências:
```bash
pip install -r requirements.txt
```

2. Execute em um único comando:
```bash
python rodar_local.py
```

3. Acesse `http://127.0.0.1:8080` no navegador e teste com `sample_data/PI_train.csv` e `sample_data/PI_test.csv` (ou `sample_data/treino.csv` e `sample_data/teste.csv`).

Localmente não é preciso definir `API_TOKEN` (sem a variável, o servidor não exige token). Para conferir a instalação local: `python scripts/verificar_deploy.py` (usa `http://127.0.0.1:8080`).

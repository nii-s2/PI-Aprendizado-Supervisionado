(() => {
  const $ = (id) => document.getElementById(id);
  const DEFAULT_HINT = "Arraste um .csv ou clique para escolher";

  const chaveEl = $("chave");
  const errorEl = $("error");
  const pill = $("pill");
  const resetBtn = $("reset");
  const trainBtn = $("btn-train");
  const testBtn = $("btn-test");

  let modeloTreinado = false;
  const pipelineSteps = Array.from(document.querySelectorAll(".pipeline li"));

  function setPipeline(doneCount, activeStep = null) {
    pipelineSteps.forEach((li, idx) => {
      const stepNum = idx + 1;
      li.classList.toggle("is-done", stepNum <= doneCount);
      li.classList.toggle("is-active", activeStep === stepNum && stepNum > doneCount);
    });
  }

  // ---------- utilidades ----------
  const fmt = (n, digits = 4) => (n === null || n === undefined ? "–" :
    Number(n).toLocaleString("pt-BR", { maximumFractionDigits: digits }));
  const fmtSinal = (n) => (n > 0 ? "+" : "") + fmt(n);

  function setError(msg) {
    errorEl.textContent = msg || "";
    errorEl.hidden = !msg;
  }

  function setBusy(btn, statusEl, busy, text) {
    btn.disabled = busy || (btn === testBtn && !modeloTreinado);
    btn.classList.toggle("busy", busy);
    statusEl.classList.toggle("busy", busy);
    statusEl.textContent = busy ? text : "";
  }

  async function api(path, options) {
    const res = await fetch(path, options);
    const data = await res.json().catch(() => ({}));
    if (!res.ok) throw new Error(data.error || "Algo deu errado ao falar com o servidor.");
    return data;
  }

  function chaveValida() {
    const raw = chaveEl.value.trim();
    if (raw === "" || !Number.isInteger(Number(raw))) {
      setError("Informe a chave de criptografia como um número inteiro.");
      chaveEl.focus();
      return false;
    }
    return true;
  }

  function renderCards(container, items, formatter = fmt) {
    container.replaceChildren(...items.map(([label, value]) => {
      const box = document.createElement("div");
      const dt = document.createElement("dt");
      const dd = document.createElement("dd");
      dt.textContent = label;
      dd.textContent = formatter(value);
      box.append(dt, dd);
      return box;
    }));
  }

  // ---------- estado do modelo ----------
  function aplicarStatus(s) {
    modeloTreinado = Boolean(s && s.treinado);
    pill.className = "pill " + (modeloTreinado ? "ok" : "off");
    pill.textContent = modeloTreinado
      ? `Treinado com ${fmt(s.amostras_treino, 0)} linhas`
      : "Não treinado";
    testBtn.disabled = !modeloTreinado;
    resetBtn.disabled = !modeloTreinado;
    if (modeloTreinado) {
      renderTreino(s);
      if ($("test-result").hidden) setPipeline(3, 4);
    } else {
      $("train-result").hidden = true;
      $("train-plot-box").hidden = true;
      $("train-downloads").hidden = true;
      const temArquivoTreino = Boolean($("train").files && $("train").files.length);
      setPipeline(0, temArquivoTreino ? 1 : null);
    }
  }

  async function atualizarStatus() {
    // no plano gratuito do Render o servidor pode estar dormindo (~1 min para acordar)
    const aviso = setTimeout(() => { pill.textContent = "Acordando o servidor… (até ~1 min)"; }, 4000);
    try {
      aplicarStatus(await api("/api/status"));
    } catch (err) {
      modeloTreinado = false;
      pill.className = "pill off";
      pill.textContent = "Servidor indisponível";
      testBtn.disabled = true;
      resetBtn.disabled = true;
    } finally {
      clearTimeout(aviso);
    }
  }

  function renderTreino(s) {
    const e = s.expectativa_desempenho;
    const t = s.metricas_treino;
    if (!e) return;

    // hold-out cronológico (últimos 20%): o "esperado" do notebook
    renderCards($("train-metrics"), [
      ["R²", e.R2_esperado], ["RMSE", e.RMSE_esperado], ["MAE", e.MAE_esperado],
      ["MSE", e.MSE_esperado], ["SSE", e.SSE_esperado], ["Linhas do hold-out", s.amostras_holdout],
    ]);

    // ajuste final na base toda: o "treino" do notebook
    if (t) {
      renderCards($("train-fit-metrics"), [
        ["R²", t.R2], ["RMSE", t.RMSE], ["MAE", t.MAE],
        ["MSE", t.MSE], ["SSE", t.SSE], ["Linhas de treino", s.amostras_treino],
      ]);
    }

    // parâmetros (np.round(params, 2) no notebook)
    $("train-params").replaceChildren(...(s.parametros || []).map((p) => {
      const tr = document.createElement("tr");
      const a = document.createElement("td");
      const b = document.createElement("td");
      a.textContent = p.nome;
      b.textContent = fmt(p.valor, 2);
      tr.append(a, b);
      return tr;
    }));

    // o gráfico só vem na resposta do treino; ao recarregar a página ele não é reenviado
    if (s.grafico_treino_png) {
      const url = `data:image/png;base64,${s.grafico_treino_png}`;
      $("plot-train").src = url;
      $("dl-plot-train").href = url;
      $("train-plot-box").hidden = false;
      $("train-downloads").hidden = false;
    }
    $("train-result").hidden = false;
  }

  // ---------- seleção de arquivos ----------
  function formatSize(bytes) {
    if (bytes < 1024) return `${bytes} B`;
    if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
    return `${(bytes / 1024 / 1024).toFixed(1)} MB`;
  }

  function showFile(input) {
    const drop = input.closest(".drop");
    const file = input.files[0];
    drop.classList.toggle("has-file", Boolean(file));
    drop.querySelector("[data-hint]").textContent =
      file ? `${file.name} (${formatSize(file.size)})` : DEFAULT_HINT;
    if (input.id === "train" && !modeloTreinado) {
      setPipeline(0, file ? 1 : null);
    }
  }

  document.querySelectorAll(".drop").forEach((drop) => {
    const input = drop.querySelector("input[type=file]");
    input.addEventListener("change", () => showFile(input));
    ["dragenter", "dragover"].forEach((ev) =>
      drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.add("is-over"); }));
    ["dragleave", "drop"].forEach((ev) =>
      drop.addEventListener(ev, () => drop.classList.remove("is-over")));
    drop.addEventListener("drop", (e) => {
      e.preventDefault();
      if (e.dataTransfer.files.length) {
        input.files = e.dataTransfer.files;
        showFile(input);
      }
    });
  });

  function montarEnvio(inputId, campo, nome) {
    const file = $(inputId).files[0];
    if (!file) { setError(`Escolha a base de ${nome}.`); return null; }
    if (!file.name.toLowerCase().endsWith(".csv")) { setError(`A base de ${nome} precisa ser um arquivo .csv.`); return null; }
    const body = new FormData();
    body.append(campo, file);
    body.append("chave", chaveEl.value.trim());
    return body;
  }

  // ---------- treino ----------
  $("form-train").addEventListener("submit", async (e) => {
    e.preventDefault();
    setError("");
    if (!chaveValida()) return;
    const body = montarEnvio("train", "train", "treino");
    if (!body) return;

    $("test-result").hidden = true;
    setPipeline(2, 3);
    setBusy(trainBtn, $("status-train"), true, "Criptografando, enviando e treinando…");
    try {
      aplicarStatus(await api("/api/treinar", { method: "POST", body }));
      setPipeline(3, 4);
    } catch (err) {
      setPipeline(0, 1);
      setError(err.message);
    } finally {
      setBusy(trainBtn, $("status-train"), false);
    }
  });

  // ---------- teste ----------
  function renderTeste(data) {
    const av = data.avaliacao;
    const msg = $("test-msg");
    msg.hidden = !av.mensagem;
    msg.textContent = av.mensagem || "";

    const real = av.desempenho_real;
    const diff = av.comparacao;
    const metricsEl = $("test-metrics");
    const diffEl = $("test-diff");
    metricsEl.previousElementSibling.hidden = !real;
    diffEl.previousElementSibling.hidden = !real;
    metricsEl.hidden = diffEl.hidden = !real;

    if (real) {
      renderCards(metricsEl, [
        ["R²", real.R2_real], ["RMSE", real.RMSE_real], ["MAE", real.MAE_real],
        ["MSE", real.MSE_real], ["SSE", real.SSE_real], ["Linhas de teste", av.amostras_teste],
      ]);
      renderCards(diffEl, [["R²", diff.diff_R2], ["MAE", diff.diff_MAE], ["MSE", diff.diff_MSE]], fmtSinal);
    }

    // gráfico só existe quando o teste tem rótulos
    const temGrafico = Boolean(data.grafico_png);
    $("plot-box").hidden = !temGrafico;
    $("dl-plot").hidden = !temGrafico;
    if (temGrafico) {
      const plotUrl = `data:image/png;base64,${data.grafico_png}`;
      $("plot").src = plotUrl;
      $("dl-plot").href = plotUrl;
    } else {
      $("plot").removeAttribute("src");
    }
    $("dl-csv").href = `data:text/csv;charset=utf-8;base64,${data.resultado_csv}`;
    $("dl-zip").href = `data:application/zip;base64,${data.resultado_zip}`;
    $("test-result").hidden = false;
    $("test-result").scrollIntoView({ behavior: "smooth", block: "start" });
  }

  $("form-test").addEventListener("submit", async (e) => {
    e.preventDefault();
    setError("");
    if (!chaveValida()) return;
    const body = montarEnvio("test", "test", "teste");
    if (!body) return;

    setPipeline(2, 3);
    setBusy(testBtn, $("status-test"), true, "Criptografando, enviando e testando…");
    try {
      renderTeste(await api("/api/testar", { method: "POST", body }));
      setPipeline(4, null);
    } catch (err) {
      setPipeline(3, 4);
      setError(err.message);
    } finally {
      setBusy(testBtn, $("status-test"), false);
    }
  });

  // ---------- reset ----------
  resetBtn.addEventListener("click", async () => {
    setError("");
    resetBtn.disabled = true;
    try {
      await api("/api/reset", { method: "POST" });
      $("test-result").hidden = true;
      aplicarStatus({ treinado: false });
    } catch (err) {
      setError(err.message);
      resetBtn.disabled = false;
    }
  });

  atualizarStatus();
})();

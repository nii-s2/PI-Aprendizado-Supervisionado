(() => {
  const $ = (id) => document.getElementById(id);
  const form = $("form");
  const submitBtn = $("submit");
  const statusEl = $("status");
  const errorEl = $("error");
  const resultsEl = $("results");

  const DEFAULT_HINT = "Arraste um .csv ou clique para escolher";

  // ---------- seleção de arquivos ----------
  function formatSize(bytes) {
    if (bytes < 1024) return `${bytes} B`;
    if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
    return `${(bytes / 1024 / 1024).toFixed(1)} MB`;
  }

  function showFile(input) {
    const drop = input.closest(".drop");
    const hint = drop.querySelector("[data-hint]");
    const file = input.files[0];
    drop.classList.toggle("has-file", Boolean(file));
    hint.textContent = file ? `${file.name} (${formatSize(file.size)})` : DEFAULT_HINT;
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

  // ---------- mensagens ----------
  function setError(msg) {
    errorEl.textContent = msg || "";
    errorEl.hidden = !msg;
  }
  function setBusy(busy) {
    submitBtn.disabled = busy;
    statusEl.classList.toggle("busy", busy);
    statusEl.textContent = busy ? "Compactando, criptografando e treinando…" : "";
  }

  // ---------- resultados ----------
  const fmt = (n) => (n === null || n === undefined ? "–" :
    Number(n).toLocaleString("pt-BR", { maximumFractionDigits: 4 }));

  function renderMetrics(m) {
    const items = [
      ["R²", m.r2], ["RMSE", m.rmse], ["MAE", m.mae], ["MSE", m.mse],
      ["Linhas de treino", m.n_train], ["Linhas de teste", m.n_test],
    ];
    $("metrics").replaceChildren(...items.map(([label, value]) => {
      const box = document.createElement("div");
      const dt = document.createElement("dt");
      const dd = document.createElement("dd");
      dt.textContent = label;
      dd.textContent = fmt(value);
      box.append(dt, dd);
      return box;
    }));

    const rows = [["Intercepto", m.intercept], ...Object.entries(m.coefficients)];
    $("coefs").replaceChildren(...rows.map(([name, value]) => {
      const tr = document.createElement("tr");
      const a = document.createElement("td");
      const b = document.createElement("td");
      a.textContent = name;
      b.textContent = fmt(value);
      tr.append(a, b);
      return tr;
    }));
  }

  function renderResults(data) {
    renderMetrics(data.metrics);
    const plotUrl = `data:image/png;base64,${data.plot_png}`;
    $("plot").src = plotUrl;
    $("dl-plot").href = plotUrl;
    $("dl-csv").href = `data:application/gzip;base64,${data.results_csv_gz}`;
    resultsEl.hidden = false;
    resultsEl.scrollIntoView({ behavior: "smooth", block: "start" });
  }

  // ---------- envio ----------
  form.addEventListener("submit", async (e) => {
    e.preventDefault();
    setError("");

    const train = $("train").files[0];
    const test = $("test").files[0];
    if (!train || !test) return setError("Escolha as duas bases: treino e teste.");
    if (![train, test].every((f) => f.name.toLowerCase().endsWith(".csv")))
      return setError("Os dois arquivos precisam ser .csv.");

    const body = new FormData();
    body.append("train", train);
    body.append("test", test);
    body.append("target", $("target").value.trim());

    resultsEl.hidden = true;
    setBusy(true);
    try {
      const res = await fetch("/api/process", { method: "POST", body });
      const data = await res.json().catch(() => ({}));
      if (!res.ok) throw new Error(data.error || "Algo deu errado ao processar as bases.");
      renderResults(data);
    } catch (err) {
      setError(err.message);
    } finally {
      setBusy(false);
    }
  });
})();

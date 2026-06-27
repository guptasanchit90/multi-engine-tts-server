document.addEventListener("DOMContentLoaded", () => {
  // ── State ────────────────────────────────────────────────────────────────
  let models = [];
  let voices = {};
  let voiceDetails = [];
  let outputs = [];
  let _modelToEngine = {};
  let filterState = { engine: '', model: '', time: '' };
  let mediaRecorder = null;
  let recordingChunks = [];

  // ── DOM refs ─────────────────────────────────────────────────────────────
  const voicePanel = document.getElementById("voice-panel");
  const outputPanel = document.getElementById("output-panel");
  const addVoiceModal = document.getElementById("add-voice-modal");

  const form = document.getElementById("tts-form");
  const genStatus = document.getElementById("gen-status");
  const modelSelect = document.getElementById("model");
  const modelDesc = document.getElementById("model-desc");
  const modelLangs = document.getElementById("model-langs");
  const textArea = document.getElementById("text");
  const speedInput = document.getElementById("speed");
  const speedValue = document.getElementById("speed-value");
  const tempInput = document.getElementById("temperature");
  const tempValue = document.getElementById("temp-value");
  const seedInput = document.getElementById("seed");
  const addPausesCheck = document.getElementById("add_pauses");
  const generateBtn = document.getElementById("generate-btn");
  const speakerName = document.getElementById("speaker_name");
  const voiceDesc = document.getElementById("voice_description");
  const voiceFile = document.getElementById("sample_voice_file");

  const voiceFields = {
    speaker: document.getElementById("voice-speaker"),
    prompt: document.getElementById("voice-prompt"),
    clone: document.getElementById("voice-clone"),
  };
  const CAP_TO_FIELD = {
    speaker: "speaker", voice_blend: "speaker",
    voice_prompt: "prompt", voice_clone: "clone",
  };

  const addVoiceBtn = document.getElementById("add-voice-btn");
  const closeModalBtn = document.getElementById("close-modal-btn");
  const uploadForm = document.getElementById("upload-form");
  const voiceFileInput = document.getElementById("voice-file");
  const voiceNameInput = document.getElementById("voice-name");
  const uploadStatus = document.getElementById("upload-status");
  const recordForm = document.getElementById("record-form");
  const voiceNameRecord = document.getElementById("voice-name-record");
  const recordBtn = document.getElementById("record-btn");
  const stopRecordBtn = document.getElementById("stop-record-btn");
  const recordStatus = document.getElementById("record-status");
  const recWave = document.getElementById("recording-wave");
  const clearOutputsBtn = document.getElementById("clear-outputs-btn");
  const filterEngine = document.getElementById("filter-engine");
  const filterModel = document.getElementById("filter-model");
  const filterTime = document.getElementById("filter-time");
  const clearFiltersBtn = document.getElementById("clear-filters-btn");
  const outputFiltersEl = document.getElementById("output-filters");

  // ── Slider displays ──────────────────────────────────────────────────────
  speedInput.addEventListener("input", () => {
    speedValue.textContent = parseFloat(speedInput.value).toFixed(2) + "x";
  });
  tempInput.addEventListener("input", () => {
    tempValue.textContent = parseFloat(tempInput.value).toFixed(1);
  });

  // ── Shared audio player state ────────────────────────────────────────────
  let _currentAudio = null;
  let _currentName = null;

  function playerHTML(item, actionsHTML) {
    const dur = item.duration ? item.duration.toFixed(1) + "s" : "";
    const size = item.size > 1048576
      ? (item.size / 1048576).toFixed(1) + " MB"
      : (item.size / 1024).toFixed(0) + " KB";
    let dateStr = "";
    if (item.created_at) {
      const d = new Date(item.created_at * 1000);
      dateStr = d.toLocaleDateString() + " " + d.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
    }
    return `<div class="player-main">
      <button class="player-play-btn" data-url="${item.url}" data-name="${item.name}" title="Play">&#9654;</button>
      <div class="player-info">
        <div class="player-title">${escapeHtml(item.name)}</div>
        <div class="player-meta">${dur}${dur && size ? " &middot; " : ""}${size}</div>
        ${dateStr ? `<div class="player-date">${dateStr}</div>` : ""}
      </div>
      <div class="player-actions">${actionsHTML}</div>
    </div>
    <div class="player-seeker" style="display:none">
      <div class="seeker-track"><div class="seeker-fill"></div></div>
      <span class="seeker-time">0:00</span>
    </div>`;
  }

  function formatTime(seconds) {
    if (!seconds || !isFinite(seconds)) return "0:00";
    const m = Math.floor(seconds / 60);
    const s = Math.floor(seconds % 60);
    return m + ":" + (s < 10 ? "0" : "") + s;
  }

  function resetPlayerUI(name) {
    const item = document.querySelector(`.item[data-name="${CSS.escape(name)}"]`);
    if (!item) return;
    const btn = item.querySelector(".player-play-btn");
    if (btn) { btn.innerHTML = "&#9654;"; btn.classList.remove("playing"); btn.title = "Play"; }
    const seeker = item.querySelector(".player-seeker");
    if (seeker) seeker.style.display = "none";
    const fill = item.querySelector(".seeker-fill");
    if (fill) fill.style.width = "0%";
    const timeEl = item.querySelector(".seeker-time");
    if (timeEl) timeEl.textContent = "0:00";
  }

  function togglePlay(btn) {
    const url = btn.dataset.url;
    const name = btn.dataset.name;
    const item = btn.closest(".item");

    if (_currentAudio && _currentName === name) {
      _currentAudio.pause(); _currentAudio = null;
      _currentName = null;
      resetPlayerUI(name);
      // Also reset this button if it's a different instance (e.g. modal)
      btn.innerHTML = "&#9654;"; btn.classList.remove("playing"); btn.title = "Play";
      const seeker = item.querySelector(".player-seeker");
      if (seeker) seeker.style.display = "none";
      return;
    }
    if (_currentAudio) {
      const prev = _currentName;
      _currentAudio.pause(); _currentAudio = null;
      _currentName = null;
      resetPlayerUI(prev);
    }
    const audio = new Audio(url);
    audio._playName = name;
    btn.innerHTML = "&#9632;";
    btn.classList.add("playing");
    btn.title = "Stop";
    const seeker = item.querySelector(".player-seeker");
    if (seeker) seeker.style.display = "";

    audio.addEventListener("timeupdate", () => {
      if (!_currentAudio) return;
      const fill = item.querySelector(".seeker-fill");
      const timeEl = item.querySelector(".seeker-time");
      if (fill && audio.duration) fill.style.width = (audio.currentTime / audio.duration * 100) + "%";
      if (timeEl && audio.duration) timeEl.textContent = formatTime(audio.currentTime) + " / " + formatTime(audio.duration);
    });

    audio.addEventListener("ended", () => {
      resetPlayerUI(name);
      _currentAudio = null;
      _currentName = null;
    });

    audio.play().catch(() => {});
    _currentAudio = audio;
    _currentName = name;
  }

  function setupSeekers(container) {
    container.querySelectorAll(".seeker-track").forEach((track) => {
      track.addEventListener("click", (e) => {
        if (!_currentAudio) return;
        const rect = track.getBoundingClientRect();
        const pct = (e.clientX - rect.left) / rect.width;
        if (_currentAudio.duration) _currentAudio.currentTime = pct * _currentAudio.duration;
      });
    });
  }

  function syncPlayerUI() {
    if (!_currentAudio || _currentAudio.paused) return;
    const item = document.querySelector(`.item[data-name="${CSS.escape(_currentName)}"]`);
    if (!item) return;
    const btn = item.querySelector(".player-play-btn");
    if (btn) { btn.innerHTML = "&#9632;"; btn.classList.add("playing"); btn.title = "Stop"; }
    const seeker = item.querySelector(".player-seeker");
    if (seeker) seeker.style.display = "";
    // Sync seeker position
    if (_currentAudio.duration) {
      const fill = item.querySelector(".seeker-fill");
      const timeEl = item.querySelector(".seeker-time");
      if (fill) fill.style.width = (_currentAudio.currentTime / _currentAudio.duration * 100) + "%";
      if (timeEl) timeEl.textContent = formatTime(_currentAudio.currentTime) + " / " + formatTime(_currentAudio.duration);
    }
  }

  // ── Initial data load ────────────────────────────────────────────────────
  function loadAll() {
    Promise.all([
      fetch("/v1/models").then((r) => r.json()),
      fetch("/voices").then((r) => r.json()),
      fetch("/voices/detail").then((r) => r.json()),
      fetch("/outputs/detail").then((r) => r.json()),
    ]).then(([modelData, voiceData, vDetails, outData]) => {
      models = Array.isArray(modelData) ? modelData : Object.values(modelData);
      _modelToEngine = {};
      models.forEach((m) => { _modelToEngine[m.id] = m.engine; });
      voices = voiceData;
      voiceDetails = vDetails;
      outputs = outData;
      populateModelSelect();
      renderVoiceList();
      renderOutputList();
      populateFilterDropdowns();
    }).catch(() => {
      genStatus.textContent = "Failed to load data. Is the server running?";
      genStatus.className = "status-msg show error";
    });
  }
  loadAll();

  // ── Model select ─────────────────────────────────────────────────────────
  function populateModelSelect() {
    modelSelect.innerHTML = '<option value="">— Select a model —</option>';
    models.forEach((m) => {
      const opt = document.createElement("option");
      opt.value = m.id;
      let label = m.name;
      if (m.capabilities) label += "  [" + m.capabilities.join(", ") + "]";
      if (!m.available) { label += " [unavailable]"; opt.disabled = true; }
      opt.textContent = label;
      modelSelect.appendChild(opt);
    });
  }

  modelSelect.addEventListener("change", () => {
    const id = modelSelect.value;
    if (!id) {
      Object.values(voiceFields).forEach((f) => f.classList.remove("active"));
      modelDesc.textContent = "";
      modelLangs.textContent = "";
      return;
    }
    const model = models.find((m) => m.id === id);
    if (!model) return;
    modelDesc.textContent = model.description || "";
    const langs = model.languages || [];
    modelLangs.textContent = langs.length ? "Languages: " + langs.join(", ") : "";
    const caps = model.capabilities || [];
    Object.values(voiceFields).forEach((f) => f.classList.remove("active"));
    for (const cap of caps) {
      const fid = CAP_TO_FIELD[cap];
      if (fid && voiceFields[fid]) { voiceFields[fid].classList.add("active"); break; }
    }
    updateVoiceHints(model);
  });

  function updateVoiceHints(model) {
    const ed = model.voices || {};
    const builtIn = ed.built_in || [];
    const cloneable = ed.cloneable || [];

    speakerName.innerHTML = '<option value="">— Select a voice —</option>';
    builtIn.forEach((v) => {
      const o = document.createElement("option"); o.value = v; o.textContent = v;
      speakerName.appendChild(o);
    });
    voiceFile.innerHTML = '<option value="">— Select a voice file —</option>';
    cloneable.forEach((v) => {
      const o = document.createElement("option"); o.value = v; o.textContent = v;
      voiceFile.appendChild(o);
    });


  }

  // ── Generate (center panel) ──────────────────────────────────────────────
  form.addEventListener("submit", async (e) => {
    e.preventDefault();
    genStatus.className = "status-msg";
    genStatus.textContent = "";

    const modelId = modelSelect.value;
    const text = textArea.value.trim();
    if (!modelId || !text) {
      genStatus.textContent = "Please select a model and enter text.";
      genStatus.className = "status-msg show error"; return;
    }
    const model = models.find((m) => m.id === modelId);
    const caps = model ? model.capabilities || [] : [];

    const body = { model: modelId, input: text, speed: parseFloat(speedInput.value) };

    if (caps.includes("speaker") || caps.includes("voice_blend")) {
      const v = speakerName.value.trim(); if (v) body.voice = v;
    } else if (caps.includes("voice_prompt")) {
      const v = voiceDesc.value.trim();
      if (!v) { genStatus.textContent = "Voice description is required."; genStatus.className = "status-msg show error"; return; }
      body.voice = v;
    } else if (caps.includes("voice_clone")) {
      const v = voiceFile.value.trim();
      if (!v) { genStatus.textContent = "A sample voice file is required."; genStatus.className = "status-msg show error"; return; }
      body.voice = v;
    }

    body.temperature = parseFloat(tempInput.value);
    const seed = seedInput.value.trim();
    if (seed) body.seed = parseInt(seed, 10);

    generateBtn.disabled = true;
    generateBtn.textContent = "Generating...";

    try {
      const resp = await fetch("/v1/audio/speech", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          "X-Save-Output": "true",
        },
        body: JSON.stringify(body),
      });
      if (!resp.ok) {
        const err = await resp.json().catch(() => ({ detail: `HTTP ${resp.status}` }));
        genStatus.textContent = err.detail || `Error ${resp.status}`;
        genStatus.className = "status-msg show error"; return;
      }
      genStatus.textContent = "Done!";
      genStatus.className = "status-msg show success";
      // Refresh output list
      const data = await fetch("/outputs/detail").then((r) => r.json());
      outputs = data;
      renderOutputList();
    } catch (err) {
      genStatus.textContent = "Network error: " + err.message;
      genStatus.className = "status-msg show error";
    } finally {
      generateBtn.disabled = false;
      generateBtn.textContent = "Generate";
    }
  });

  // ── Voice panel (left) ───────────────────────────────────────────────────
  function renderVoiceList() {
    if (!voiceDetails.length) {
      voicePanel.innerHTML = '<div class="empty">No voice files yet.<br>Click "+ Add" to upload or record one.</div>';
      return;
    }
    let html = "";
    for (const v of voiceDetails) {
      const actions = `
        <button class="icon-btn" data-rename="${v.name}" title="Rename">&#9998;</button>
        <button class="icon-btn danger" data-del="${v.name}" title="Delete">&times;</button>`;
      html += `<div class="item" data-name="${v.name}">${playerHTML(v, actions)}</div>`;
    }
    voicePanel.innerHTML = html;
    setupSeekers(voicePanel);
    syncPlayerUI();

    // Play
    voicePanel.querySelectorAll(".player-play-btn").forEach((btn) => {
      btn.addEventListener("click", () => togglePlay(btn));
    });
    // Delete
    voicePanel.querySelectorAll("[data-del]").forEach((btn) => {
      btn.addEventListener("click", async () => {
        const name = btn.dataset.del;
        if (!confirm('Delete "' + name + '"?')) return;
        const resp = await fetch("/voice/" + encodeURIComponent(name), { method: "DELETE" });
        if (!resp.ok) return;
        if (_currentAudio && _currentName === name) { _currentAudio.pause(); _currentAudio = null; _currentName = null; }
        voiceDetails = voiceDetails.filter((v) => v.name !== name);
        Object.keys(voices).forEach((eng) => {
          const cats = voices[eng];
          Object.keys(cats).forEach((cat) => {
            if (Array.isArray(cats[cat])) cats[cat] = cats[cat].filter((v) => v !== name);
          });
        });
        models.forEach((m) => {
          if (m.voices && m.voices.cloneable) {
            m.voices.cloneable = m.voices.cloneable.filter((v) => v !== name);
          }
        });
        renderVoiceList();
      });
    });
    // Rename
    voicePanel.querySelectorAll("[data-rename]").forEach((btn) => {
      btn.addEventListener("click", () => startRename(btn.dataset.rename));
    });
  }

  function startRename(name) {
    const item = voicePanel.querySelector(`[data-name="${name}"]`);
    if (!item) return;
    const nameEl = item.querySelector(".player-title");
    const current = nameEl.textContent;
    const input = document.createElement("input");
    input.type = "text";
    input.className = "name-edit";
    input.value = current;
    input.dataset.original = current;
    nameEl.replaceWith(input);
    input.focus();
    input.select();

    function submitRename() {
      const val = input.value.trim();
      if (!val || val === current) {
        const span = document.createElement("div");
        span.className = "player-title";
        span.textContent = current;
        input.replaceWith(span);
        return;
      }
      fetch("/voice/" + encodeURIComponent(current) + "?new_name=" + encodeURIComponent(val), { method: "PUT" })
        .then((r) => {
          if (!r.ok) { alert("Rename failed"); throw new Error(); }
          return r.json();
        })
        .then((data) => {
          const v = voiceDetails.find((v) => v.name === current);
          if (v) { v.name = data.name; v.url = data.url; }
          renderVoiceList();
          // Update voices cache
          Object.keys(voices).forEach((eng) => {
            const cats = voices[eng];
            Object.keys(cats).forEach((cat) => {
              if (Array.isArray(cats[cat])) {
                const idx = cats[cat].indexOf(current);
                if (idx >= 0) cats[cat][idx] = data.name;
              }
            });
          });
          models.forEach((m) => {
            if (m.voices && m.voices.cloneable) {
              const idx = m.voices.cloneable.indexOf(current);
              if (idx >= 0) m.voices.cloneable[idx] = data.name;
            }
          });
        })
        .catch(() => {
          const span = document.createElement("div");
          span.className = "player-title";
          span.textContent = current;
          input.replaceWith(span);
        });
    }

    input.addEventListener("blur", submitRename);
    input.addEventListener("keydown", (e) => {
      if (e.key === "Enter") { e.preventDefault(); input.blur(); }
      if (e.key === "Escape") { input.value = current; input.blur(); }
    });
  }

  // ── Filter helpers ──────────────────────────────────────────────────────
  function applyFilters(items) {
    return items.filter((o) => {
      const eng = _modelToEngine[o.params?.model] || "";
      if (filterState.engine && eng !== filterState.engine) return false;
      if (filterState.model && o.params?.model !== filterState.model) return false;
      if (filterState.time) {
        const cutoff = Date.now() / 1000 - filterState.time * 60;
        if ((o.created_at || 0) < cutoff) return false;
      }
      return true;
    });
  }

  function populateFilterDropdowns() {
    const hasOutputs = outputs.length > 0;
    outputFiltersEl.style.display = hasOutputs ? "" : "none";
    if (!hasOutputs) return;

    const engines = new Set();
    const models = new Set();
    for (const o of outputs) {
      const eng = _modelToEngine[o.params?.model];
      if (eng) engines.add(eng);
      if (o.params?.model) models.add(o.params.model);
    }

    const selEngine = filterEngine.value;
    const selModel = filterModel.value;
    filterEngine.innerHTML = '<option value="">Engine</option>';
    filterModel.innerHTML = '<option value="">Model</option>';
    for (const e of [...engines].sort()) {
      const opt = document.createElement("option");
      opt.value = e; opt.textContent = e;
      if (e === selEngine) opt.selected = true;
      filterEngine.appendChild(opt);
    }
    for (const m of [...models].sort()) {
      const opt = document.createElement("option");
      opt.value = m; opt.textContent = m;
      if (m === selModel) opt.selected = true;
      filterModel.appendChild(opt);
    }
  }

  // ── Output panel (right) ─────────────────────────────────────────────────
  function renderOutputList() {
    populateFilterDropdowns();
    if (!outputs.length) {
      outputPanel.innerHTML = '<div class="empty">No generated outputs yet.<br>Use the Generate panel to create speech.</div>';
      return;
    }
    let html = "";
    const filtered = applyFilters(outputs);
    if (!filtered.length) {
      outputPanel.innerHTML = '<div class="empty">No outputs match the current filters.</div>';
      return;
    }
    for (const o of filtered) {
      const actions = `
        <button class="icon-btn" data-toggle="${o.name}" title="Show details">&#9432;</button>
        <button class="icon-btn danger" data-del-out="${o.name}" title="Delete">&times;</button>`;
      html += `<div class="item" data-name="${o.name}">${playerHTML(o, actions)}</div>`;
    }
    outputPanel.innerHTML = html;
    setupSeekers(outputPanel);
    syncPlayerUI();

    // Play
    outputPanel.querySelectorAll(".player-play-btn").forEach((btn) => {
      btn.addEventListener("click", () => togglePlay(btn));
    });
    // Delete
    outputPanel.querySelectorAll("[data-del-out]").forEach((btn) => {
      btn.addEventListener("click", async () => {
        const name = btn.dataset.delOut;
        if (!confirm('Delete "' + name + '"?')) return;
        const resp = await fetch("/output/" + encodeURIComponent(name), { method: "DELETE" });
        if (!resp.ok) return;
        if (_currentAudio && _currentName === name) { _currentAudio.pause(); _currentAudio = null; _currentName = null; }
        outputs = outputs.filter((o) => o.name !== name);
        renderOutputList();
      });
    });
    // Show details modal
    outputPanel.querySelectorAll("[data-toggle]").forEach((btn) => {
      btn.addEventListener("click", () => {
        const name = btn.dataset.toggle;
        const out = outputs.find((o) => o.name === name);
        if (!out || !out.params) return;
        _modalOut = out;
        _regenParams = out.params;

        // Player section in modal — same shared component
        const downloadBtn = `<a class="player-dl" href="${out.url}" download title="Download">&#x2193;</a>`;
        const deleteBtn = `<button class="icon-btn danger modal-del-btn" title="Delete" data-name="${out.name}">&times;</button>`;
        const player = `<div class="item" data-name="${out.name}">${playerHTML(out, downloadBtn + deleteBtn)}</div>`;
        document.getElementById("params-player").innerHTML = player;

        // Params section — read-only display
        const p = out.params;
        const keys = Object.keys(p);
        let html = "";
        for (const k of keys) {
          const val = String(p[k] ?? "");
          const isLong = val.length > 80;
          html += `<div class="params-field">
            <div class="params-label">${k}</div>
            <div class="params-value${isLong ? " long-text" : ""}">${escapeHtml(val)}</div>
          </div>`;
        }
        document.getElementById("params-content").innerHTML = html;

        document.getElementById("params-regen-status").className = "status-msg";
        document.getElementById("params-modal").classList.add("show");
        syncPlayerUI();

        // Wire play in modal
        setupSeekers(document.getElementById("params-modal"));
        const modalBtn = document.querySelector("#params-player .player-play-btn");
        if (modalBtn) modalBtn.addEventListener("click", () => togglePlay(modalBtn));
        // Wire delete in modal
        document.querySelector("#params-player .modal-del-btn")?.addEventListener("click", async () => {
          if (!confirm('Delete "' + out.name + '"?')) return;
          const resp = await fetch("/output/" + encodeURIComponent(out.name), { method: "DELETE" });
          if (!resp.ok) return;
          if (_currentAudio && _currentName === out.name) { _currentAudio.pause(); _currentAudio = null; _currentName = null; }
          outputs = outputs.filter((o) => o.name !== out.name);
          renderOutputList();
          document.getElementById("params-modal").classList.remove("show");
        });
      });
    });
  }

  // Clear outputs
  clearOutputsBtn.addEventListener("click", async () => {
    if (!confirm("Delete all generated outputs?")) return;
    const resp = await fetch("/outputs", { method: "DELETE" });
    if (!resp.ok) return;
    if (_currentAudio) { _currentAudio.pause(); _currentAudio = null; _currentName = null; }
    outputs = [];
    renderOutputList();
  });

  // ── Filter event listeners ───────────────────────────────────────────────
  function applyFilter() {
    filterState.engine = filterEngine.value;
    filterState.model = filterModel.value;
    filterState.time = filterTime.value;
    renderOutputList();
  }

  filterEngine.addEventListener("change", applyFilter);
  filterModel.addEventListener("change", applyFilter);
  filterTime.addEventListener("change", applyFilter);

  clearFiltersBtn.addEventListener("click", () => {
    filterEngine.value = "";
    filterModel.value = "";
    filterTime.value = "";
    filterState = { engine: '', model: '', time: '' };
    renderOutputList();
  });

  // ── State for modal ──────────────────────────────────────────────────────
  let _regenParams = null;
  let _modalOut = null;

  // ── Utils ─────────────────────────────────────────────────────────────────
  function escapeHtml(s) {
    const d = document.createElement("div");
    d.textContent = s;
    return d.innerHTML;
  }

  // ── Regenerate ───────────────────────────────────────────────────────────
  function fillForm(params) {
    if (!params) return;
    if (params.model) {
      const opt = Array.from(modelSelect.options).find((o) => o.value === params.model);
      if (opt) { modelSelect.value = params.model; modelSelect.dispatchEvent(new Event("change")); }
    }
    if (params.input) textArea.value = params.input;
    if (params.speed != null) {
      const s = Math.min(4, Math.max(0.25, parseFloat(params.speed)));
      speedInput.value = s.toString();
      speedValue.textContent = s.toFixed(2) + "x";
    }
    seedInput.value = params.seed != null ? params.seed : "";
    const model = models.find((m) => m.id === params.model);
    if (model && params.voice) {
      const caps = model.capabilities || [];
      if (caps.includes("speaker") || caps.includes("voice_blend")) speakerName.value = params.voice;
      else if (caps.includes("voice_prompt")) voiceDesc.value = params.voice;
      else if (caps.includes("voice_clone")) voiceFile.value = params.voice;
    }
  }

  document.getElementById("regen-btn").addEventListener("click", () => {
    if (_regenParams) { fillForm(_regenParams); paramsModal.classList.remove("show"); }
  });

  // ── Add Voice Modal ──────────────────────────────────────────────────────
  addVoiceBtn.addEventListener("click", () => addVoiceModal.classList.add("show"));
  closeModalBtn.addEventListener("click", () => addVoiceModal.classList.remove("show"));
  addVoiceModal.addEventListener("click", (e) => {
    if (e.target === addVoiceModal) addVoiceModal.classList.remove("show");
  });

  // ── Params Modal ─────────────────────────────────────────────────────────
  const paramsModal = document.getElementById("params-modal");
  const closeParamsBtn = document.getElementById("close-params-btn");
  closeParamsBtn.addEventListener("click", () => paramsModal.classList.remove("show"));
  paramsModal.addEventListener("click", (e) => {
    if (e.target === paramsModal) paramsModal.classList.remove("show");
  });
  document.addEventListener("keydown", (e) => {
    if (e.key === "Escape") {
      paramsModal.classList.remove("show");
      addVoiceModal.classList.remove("show");
    }
  });

  // Modal tab switching
  document.querySelectorAll(".modal-tab").forEach((btn) => {
    btn.addEventListener("click", () => {
      document.querySelectorAll(".modal-tab").forEach((b) => b.classList.remove("active"));
      document.querySelectorAll(".modal-tab-content").forEach((c) => c.classList.remove("active"));
      btn.classList.add("active");
      document.getElementById("mtab-" + btn.dataset.mtab).classList.add("active");
    });
  });

  // ── Upload tab ───────────────────────────────────────────────────────────
  uploadForm.addEventListener("submit", async (e) => {
    e.preventDefault();
    uploadStatus.className = "status-msg";
    const file = voiceFileInput.files[0];
    if (!file) {
      uploadStatus.textContent = "Please select a file.";
      uploadStatus.className = "status-msg show error"; return;
    }
    const btn = document.getElementById("upload-btn");
    btn.disabled = true; btn.textContent = "Uploading...";
    try {
      const fd = new FormData();
      fd.append("file", file);
      const customName = voiceNameInput.value.trim();
      if (customName) fd.append("name", customName);
      const resp = await fetch("/voice", { method: "POST", body: fd });
      const data = await resp.json();
      if (!resp.ok) {
        uploadStatus.textContent = data.detail || `Error ${resp.status}`;
        uploadStatus.className = "status-msg show error"; return;
      }
      voiceDetails.unshift(data);
      uploadStatus.textContent = 'Uploaded "' + data.name + '" (' + (data.duration || "?") + "s)";
      uploadStatus.className = "status-msg show success";
      voiceFileInput.value = "";
      voiceNameInput.value = "";
      renderVoiceList();
      fetch("/voices").then((r) => r.json()).then((d) => { voices = d; }).catch(() => {});
      // Refresh model hints — push new voice into current model
      const sel = modelSelect.value;
      if (sel) {
        const m = models.find((mdl) => mdl.id === sel);
        if (m) {
          const v = data.name;
          if (m.voices && m.voices.cloneable && !m.voices.cloneable.includes(v)) {
            m.voices.cloneable.push(v);
          }
          updateVoiceHints(m);
        }
      }
    } catch (err) {
      uploadStatus.textContent = "Network error: " + err.message;
      uploadStatus.className = "status-msg show error";
    } finally {
      btn.disabled = false; btn.textContent = "Upload";
    }
  });

  // ── Record tab ───────────────────────────────────────────────────────────
  let recCtx = null;
  let recAnalyser = null;
  let recAnimId = null;

  recordBtn.addEventListener("click", async () => {
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
      recordStatus.textContent = window.isSecureContext === false
        ? "Recording requires HTTPS / localhost. Use http://localhost:8000."
        : "Recording not supported in this browser.";
      recordStatus.className = "status-msg show error"; return;
    }
    const name = voiceNameRecord.value.trim();
    if (!name) {
      recordStatus.textContent = "Please enter a name for the recording.";
      recordStatus.className = "status-msg show error"; return;
    }
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      recordingChunks = [];
      mediaRecorder = new MediaRecorder(stream, { mimeType: MediaRecorder.isTypeSupported("audio/webm;codecs=opus") ? "audio/webm;codecs=opus" : "audio/webm" });
      mediaRecorder.ondataavailable = (e) => { if (e.data.size > 0) recordingChunks.push(e.data); };
      mediaRecorder.onstop = async () => {
        stream.getTracks().forEach((t) => t.stop());
        if (recAnimId) { cancelAnimationFrame(recAnimId); recAnimId = null; }
        recWave.style.display = "none";

        const blob = new Blob(recordingChunks, { type: mediaRecorder.mimeType });
        const fd = new FormData();
        fd.append("file", blob, name + ".webm");
        fd.append("name", name);
        recordStatus.textContent = "Uploading...";
        recordStatus.className = "status-msg show";
        const resp = await fetch("/voice", { method: "POST", body: fd });
        const data = await resp.json();
        if (!resp.ok) {
          recordStatus.textContent = data.detail || `Error ${resp.status}`;
          recordStatus.className = "status-msg show error"; return;
        }
        voiceDetails.unshift(data);
        recordStatus.textContent = 'Uploaded "' + data.name + '" (' + (data.duration || "?") + "s)";
        recordStatus.className = "status-msg show success";
        voiceNameRecord.value = "";
        renderVoiceList();
        fetch("/voices").then((r) => r.json()).then((d) => { voices = d; }).catch(() => {});
        // Push new voice into current model
        const sel = modelSelect.value;
        if (sel) {
          const m = models.find((mdl) => mdl.id === sel);
          if (m) {
            const v = data.name;
            if (m.voices && m.voices.cloneable && !m.voices.cloneable.includes(v)) {
              m.voices.cloneable.push(v);
            }
            updateVoiceHints(m);
          }
        }
      };
      mediaRecorder.start();
      recordBtn.style.display = "none";
      stopRecordBtn.style.display = "";
      recordStatus.textContent = "Recording... Speak now.";
      recordStatus.className = "status-msg show success";

      // Visualizer
      recCtx = new AudioContext();
      const src = recCtx.createMediaStreamSource(stream);
      recAnalyser = recCtx.createAnalyser();
      recAnalyser.fftSize = 128;
      src.connect(recAnalyser);
      recWave.style.display = "block";
      drawWave();
    } catch (err) {
      recordStatus.textContent = "Microphone access denied: " + err.message;
      recordStatus.className = "status-msg show error";
    }
  });

  stopRecordBtn.addEventListener("click", () => {
    if (mediaRecorder && mediaRecorder.state !== "inactive") {
      mediaRecorder.stop();
    }
    recordBtn.style.display = "";
    stopRecordBtn.style.display = "none";
  });

  function drawWave() {
    if (!recAnalyser) return;
    const canvas = recWave;
    const ctx = canvas.getContext("2d");
    const bufferLength = recAnalyser.frequencyBinCount;
    const dataArray = new Uint8Array(bufferLength);
    const w = canvas.width, h = canvas.height;

    function draw() {
      if (!recAnalyser) return;
      recAnimId = requestAnimationFrame(draw);
      recAnalyser.getByteTimeDomainData(dataArray);
      ctx.fillStyle = "#0d1117";
      ctx.fillRect(0, 0, w, h);
      ctx.lineWidth = 2;
      ctx.strokeStyle = "#58a6ff";
      ctx.beginPath();
      const sliceWidth = w / bufferLength;
      let x = 0;
      for (let i = 0; i < bufferLength; i++) {
        const v = dataArray[i] / 128.0;
        const y = v * h / 2;
        if (i === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
        x += sliceWidth;
      }
      ctx.lineTo(w, h / 2);
      ctx.stroke();
    }
    draw();
  }

  // ── Curl Console ──────────────────────────────────────────────────────────
  function buildCurlCommand() {
    const modelId = modelSelect.value;
    const text = textArea.value.trim();
    if (!modelId || !text) return null;

    const model = models.find((m) => m.id === modelId);
    const caps = model ? model.capabilities || [] : [];

    const body = { model: modelId, input: text, speed: parseFloat(speedInput.value) };

    if (caps.includes("speaker") || caps.includes("voice_blend")) {
      const v = speakerName.value.trim();
      if (v) body.voice = v;
    } else if (caps.includes("voice_prompt")) {
      const v = voiceDesc.value.trim();
      if (v) body.voice = v;
    } else if (caps.includes("voice_clone")) {
      const v = voiceFile.value.trim();
      if (v) body.voice = v;
    }

    body.temperature = parseFloat(tempInput.value);
    const seed = seedInput.value.trim();
    if (seed) body.seed = parseInt(seed, 10);

    const json = JSON.stringify(body, null, 2);
    const escaped = json.replace(/'/g, "'\\''");

    return [
      "curl -X POST http://localhost:8000/v1/audio/speech \\",
      '  -H "Content-Type: application/json" \\',
      '  -H "X-Save-Output: true" \\',
      "  -d '" + escaped + "' \\",
      "  --output speech.mp3",
    ].join("\n");
  }

  function updateCurlDisplay() {
    const cmd = buildCurlCommand();
    const output = document.getElementById("curl-output");
    const badge = document.getElementById("curl-ready-badge");
    if (cmd) {
      output.innerHTML = escapeHtml(cmd);
      badge.textContent = "ready";
      badge.style.borderColor = "rgba(63,185,80,0.3)";
      badge.style.background = "rgba(63,185,80,0.12)";
      badge.style.color = "var(--success)";
    } else {
      output.innerHTML = '<span class="curl-placeholder"># Fill out the form to generate a curl command...</span>';
      badge.textContent = "waiting";
      badge.style.borderColor = "";
      badge.style.background = "";
      badge.style.color = "";
    }
  }

  // Wire form fields
  const curlFields = [textArea, modelSelect, speakerName, voiceDesc, voiceFile, speedInput, tempInput, seedInput, addPausesCheck];
  curlFields.forEach((el) => {
    el.addEventListener("input", updateCurlDisplay);
    el.addEventListener("change", updateCurlDisplay);
  });

  // Toggle console
  const curlToggle = document.getElementById("curl-toggle");
  const curlBody = document.getElementById("curl-body");
  curlToggle.addEventListener("click", () => {
    const open = curlBody.classList.toggle("open");
    curlToggle.classList.toggle("open");
    curlToggle.querySelector(".curl-label").textContent = open ? "Hide curl" : "Show curl";
  });

  // Copy to clipboard
  document.getElementById("curl-copy").addEventListener("click", async () => {
    const text = document.getElementById("curl-output").textContent;
    if (!text || text.startsWith("#")) return;
    try {
      await navigator.clipboard.writeText(text);
      const btn = document.getElementById("curl-copy");
      btn.textContent = "Copied!";
      setTimeout(() => { btn.textContent = "Copy"; }, 2000);
    } catch {
      // fallback
      const ta = document.createElement("textarea");
      ta.value = text;
      document.body.appendChild(ta);
      ta.select();
      document.execCommand("copy");
      document.body.removeChild(ta);
    }
  });
});

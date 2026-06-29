(function () {
const comps = window.__comps = {};

// ── Audio Player ────────────────────────────────────────────────────────────
comps['audio-player'] = {
  template: '#audio-player-template',
  props: {
    item: { type: Object, required: true },
    showRename: Boolean,
    showDelete: Boolean,
    showDetail: Boolean,
    showDownload: Boolean,
    primaryLabel: String,
  },
  emits: ['delete', 'rename', 'detail'],
  data() {
    return {
      playing: false,
      currentTime: 0,
      duration: 0,
      seeker: false,
      _audio: null,
    };
  },
  computed: {
    durStr() {
      return this.item.duration ? this.item.duration.toFixed(1) + 's' : '';
    },
    sizeStr() {
      if (!this.item.size) return '';
      return this.item.size > 1048576
        ? (this.item.size / 1048576).toFixed(1) + ' MB'
        : (this.item.size / 1024).toFixed(0) + ' KB';
    },
    dateStr() {
      if (!this.item.created_at) return '';
      const d = new Date(this.item.created_at * 1000);
      return d.toLocaleDateString() + ' ' + d.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
    },
    isCurrent() {
      return this.$store.currentName === this.item.name && this.playing;
    },
  },
  watch: {
    '$store.currentName'(name) {
      if (name !== this.item.name && this.playing) this.stop();
    },
  },
  unmounted() {
    this.stop();
  },
  methods: {
    formatTime(s) {
      if (!s || !isFinite(s)) return '0:00';
      const m = Math.floor(s / 60);
      const sec = Math.floor(s % 60);
      return m + ':' + (sec < 10 ? '0' : '') + sec;
    },
    togglePlay() {
      if (this._audio && this.playing) {
        this.stop();
        return;
      }
      // Stop any other playing audio
      if (this.$store.currentAudio) {
        this.$store.currentAudio.pause();
        this.$store.currentAudio = null;
        this.$store.currentName = '';
      }
      const audio = new Audio(this.item.url);
      this._audio = audio;
      this.seeker = true;
      this.playing = true;
      this.$store.currentAudio = audio;
      this.$store.currentName = this.item.name;

      audio.addEventListener('timeupdate', () => {
        this.currentTime = audio.currentTime;
        this.duration = audio.duration || 0;
      });
      audio.addEventListener('ended', () => this.stop());
      audio.play().catch(() => this.stop());
    },
    stop() {
      if (this._audio) {
        this._audio.pause();
        this._audio = null;
      }
      this.playing = false;
      this.seeker = false;
      this.currentTime = 0;
      this.duration = 0;
      if (this.$store.currentName === this.item.name) {
        this.$store.currentAudio = null;
        this.$store.currentName = '';
      }
    },
    seek(e) {
      const rect = e.currentTarget.getBoundingClientRect();
      const pct = (e.clientX - rect.left) / rect.width;
      const audio = this._audio || this.$store.currentAudio;
      if (audio && audio.duration) audio.currentTime = pct * audio.duration;
    },
  },
};

// ── Voice Panel ──────────────────────────────────────────────────────────────
comps['voice-panel'] = {
  template: '#voice-panel-template',
  data() {
    return { renaming: null };
  },
  computed: {
    items() { return this.$store.voiceDetails; },
  },
  methods: {
    startRename(name) {
      this.renaming = { original: name, current: name };
      this.$nextTick(() => {
        const el = this.$el.querySelector('.name-edit');
        if (el) { el.focus(); el.select(); }
      });
    },
    submitRename() {
      const r = this.renaming;
      if (!r) return;
      const val = r.current.trim();
      if (!val || val === r.original) { this.renaming = null; return; }
      fetch('/voice/' + encodeURIComponent(r.original) + '?new_name=' + encodeURIComponent(val), { method: 'PUT' })
        .then(r2 => r2.json())
        .then(data => {
          const v = this.$store.voiceDetails.find(v => v.name === r.original);
          if (v) { v.name = data.name; v.url = data.url; }
          // Update cloneable lists
          const oldName = r.original;
          this.$store.models.forEach(m => {
            if (m.voices && m.voices.cloneable) {
              const idx = m.voices.cloneable.indexOf(oldName);
              if (idx >= 0) m.voices.cloneable[idx] = data.name;
            }
          });
          this.renaming = null;
        })
        .catch(() => { this.renaming = null; });
    },
    cancelRename() { this.renaming = null; },
    deleteVoice(name) {
      if (!confirm('Delete "' + name + '"?')) return;
      fetch('/voice/' + encodeURIComponent(name), { method: 'DELETE' })
        .then(r => {
          if (!r.ok) return;
          if (this.$store.currentName === name) {
            if (this.$store.currentAudio) this.$store.currentAudio.pause();
            this.$store.currentAudio = null;
            this.$store.currentName = '';
          }
          this.$store.voiceDetails = this.$store.voiceDetails.filter(v => v.name !== name);
          Object.keys(this.$store.voices).forEach(eng => {
            const cats = this.$store.voices[eng];
            Object.keys(cats).forEach(cat => {
              if (Array.isArray(cats[cat])) cats[cat] = cats[cat].filter(v => v !== name);
            });
          });
          this.$store.models.forEach(m => {
            if (m.voices && m.voices.cloneable) m.voices.cloneable = m.voices.cloneable.filter(v => v !== name);
          });
        });
    },
  },
};

// ── Output Panel ─────────────────────────────────────────────────────────────
comps['output-panel'] = {
  template: '#output-panel-template',
  computed: {
    hasOutputs() { return this.$store.outputs.length > 0; },
    engineOptions() {
      const s = new Set();
      this.$store.outputs.forEach(o => {
        const eng = this.$store.models.find(m => m.id === (o.params && o.params.model));
        if (eng) s.add(eng.engine);
      });
      return [...s].sort();
    },
    modelOptions() {
      const s = new Set();
      this.$store.outputs.forEach(o => {
        if (o.params && o.params.model) s.add(o.params.model);
      });
      return [...s].sort();
    },
    filtered() {
      const fs = this.$store.filterState;
      return this.$store.outputs.filter(o => {
        const eng = this.$store.models.find(m => m.id === (o.params && o.params.model));
        if (fs.engine && eng && eng.engine !== fs.engine) return false;
        if (fs.model && o.params && o.params.model !== fs.model) return false;
        if (fs.time) {
          const cutoff = Date.now() / 1000 - parseInt(fs.time) * 60;
          if ((o.created_at || 0) < cutoff) return false;
        }
        return true;
      });
    },
    displayItems() {
      const list = this.filtered;
      const groups = {};
      const individual = [];
      list.forEach(o => {
        const bid = o.params && o.params.batch_id;
        if (bid) {
          if (!groups[bid]) groups[bid] = { batchId: bid, items: [], _maxTs: 0 };
          groups[bid].items.push(o);
          if (o.created_at > groups[bid]._maxTs) groups[bid]._maxTs = o.created_at;
        } else {
          individual.push(o);
        }
      });
      const result = [];
      Object.keys(groups).forEach(k => {
        const g = groups[k];
        g.items.sort((a, b) =>
          (a.params && a.params.batch_seq != null ? a.params.batch_seq : 0) -
          (b.params && b.params.batch_seq != null ? b.params.batch_seq : 0)
        );
        const names = g.items.map(x => {
          const m = this.$store.models.find(mdl => mdl.id === (x.params && x.params.model));
          return m ? m.name : (x.params && x.params.model) || '?';
        });
        result.push({ _type: 'batch_summary', batchId: k, label: names.join(', '), items: g.items, _ts: g._maxTs });
      });
      individual.forEach(o => result.push(o));
      result.sort((a, b) => (b._ts || b.created_at || 0) - (a._ts || a.created_at || 0));
      return result;
    },
  },
  methods: {
    clearFilters() {
      this.$store.filterState = { engine: '', model: '', time: '' };
    },
    clearOutputs() {
      if (!confirm('Delete all generated outputs?')) return;
      fetch('/outputs', { method: 'DELETE' }).then(r => {
        if (!r.ok) return;
        if (this.$store.currentAudio) { this.$store.currentAudio.pause(); this.$store.currentAudio = null; this.$store.currentName = ''; }
        this.$store.outputs = [];
      });
    },
    deleteOutput(name) {
      if (!confirm('Delete "' + name + '"?')) return;
      fetch('/output/' + encodeURIComponent(name), { method: 'DELETE' }).then(r => {
        if (!r.ok) return;
        if (this.$store.currentName === name) { this.$store.currentAudio.pause(); this.$store.currentAudio = null; this.$store.currentName = ''; }
        this.$store.outputs = this.$store.outputs.filter(o => o.name !== name);
      });
    },
    showDetail(out) {
      this.$store.modalOut = out;
      this.$store.regenParams = out.params || null;
      this.$store.showParams = true;
    },
    openBatchDetail(group) {
      this.$store.batchDetailItems = group.items;
      this.$store.batchDetailLabel = group.label;
      this.$store.showBatchDetail = true;
    },
    batchTotalStr(group) {
      let total = 0;
      group.items.forEach(o => { if (o.duration) total += o.duration; });
      return total.toFixed(1) + 's total';
    },
  },
};

// ── Generate Form ────────────────────────────────────────────────────────────
comps['generate-form'] = {
  template: '#generate-form-template',
  data() {
    return {
      genStatus: '',
      genStatusClass: '',
      btnState: '',
    };
  },
  computed: {
    selectedModel() {
      return this.$store.models.find(m => m.id === this.$store.form.model) || null;
    },
    capabilities() {
      return (this.selectedModel && this.selectedModel.capabilities) || [];
    },
    modelVoices() {
      const m = this.selectedModel;
      if (!m || !m.voices) return { built_in: [], cloneable: [] };
      return m.voices;
    },
    modelEngine() {
      return (this.selectedModel && this.selectedModel.engine) || '';
    },
    modelDesc() {
      return (this.selectedModel && this.selectedModel.description) || '';
    },
    modelLangs() {
      return (this.selectedModel && this.selectedModel.languages) || [];
    },
    groupedModels() {
      const groups = {};
      this.$store.models.forEach(m => {
        const e = m.engine || 'other';
        if (!groups[e]) groups[e] = [];
        groups[e].push(m);
      });
      return groups;
    },
    engineOrder() {
      return ['qwen', 'kokoro', 'piper', 'chatterbox'];
    },
    sortedEngines() {
      return Object.keys(this.groupedModels).sort(
        (a, b) => {
          const o = this.engineOrder;
          return (o.indexOf(a) === -1 ? 99 : o.indexOf(a)) - (o.indexOf(b) === -1 ? 99 : o.indexOf(b));
        }
      );
    },
    curlCommand() {
      const m = this.$store.form.model;
      const t = this.$store.form.text.trim();
      if (!m || !t) return null;
      const caps = this.capabilities;
      const body = { model: m, input: t, speed: parseFloat(this.$store.form.speed) };
      if (caps.includes('speaker') || caps.includes('voice_blend')) {
        const v = this.$store.form.speaker_name.trim();
        if (v) body.voice = v;
      } else if (caps.includes('voice_prompt')) {
        const v = this.$store.form.voice_description.trim();
        if (v) body.voice = v;
      } else if (caps.includes('voice_clone')) {
        const v = this.$store.form.sample_voice_file.trim();
        if (v) body.voice = v;
      }
      body.temperature = parseFloat(this.$store.form.temperature);
      if (this.$store.form.seed) body.seed = parseInt(this.$store.form.seed, 10);
      body.add_pauses = this.$store.form.add_pauses;
      const json = JSON.stringify(body, null, 2);
      return [
        'curl -X POST http://localhost:8000/v1/audio/speech \\',
        '  -H "Content-Type: application/json" \\',
        '  -H "X-Save-Output: true" \\',
        "  -d '" + json.replace(/'/g, "'\\''") + "' \\",
        '  --output speech.mp3',
      ].join('\n');
    },
    curlReady() { return !!this.curlCommand; },

    // Batch mode computed
    batchTabs() {
      const tabs = [];
      if (this.cloneModels.length) tabs.push({ key: 'clone', label: 'Voice Clone' });
      if (this.promptModels.length) tabs.push({ key: 'prompt', label: 'Voice Prompt' });
      if (this.speakerModels.length) tabs.push({ key: 'multivoice', label: 'Multi-Voice' });
      return tabs;
    },
    cloneModels() {
      return this.$store.models.filter(m => m.available && m.capabilities.includes('voice_clone'));
    },
    promptModels() {
      return this.$store.models.filter(m => m.available && m.capabilities.includes('voice_prompt'));
    },
    speakerModels() {
      return this.$store.models.filter(m => m.available && (m.capabilities.includes('speaker') || m.capabilities.includes('voice_blend')));
    },
    speakerModelsByEngine() {
      const groups = {};
      this.speakerModels.forEach(m => {
        const e = m.engine || 'other';
        if (!groups[e]) groups[e] = [];
        groups[e].push(m);
      });
      return groups;
    },
    speakerEngineNames() {
      return Object.keys(this.speakerModelsByEngine);
    },
    multivoiceVoices() {
      const m = this.$store.models.find(x => x.id === this.$store.multivoiceModel);
      return (m && m.voices && m.voices.built_in) ? m.voices.built_in : [];
    },
    multivoiceSelectedCount() {
      return this.$store.multivoiceSelectedVoices.length;
    },
    allCloneableVoices() {
      return this.$store.voiceDetails.map(v => v.name);
    },
    cloneBatchLabel() {
      const n = this.$store.selectedBatchModels.length;
      const total = this.cloneModels.length;
      return n + ' of ' + total + ' selected';
    },
    promptBatchLabel() {
      const n = this.$store.selectedBatchModels.length;
      const total = this.promptModels.length;
      return n + ' of ' + total + ' selected';
    },
    batchSubmitDisabled() {
      if (!this.$store.batchMode) return false;
      if (this.$store.batchTab === 'multivoice') {
        return !this.$store.multivoiceModel || !this.$store.multivoiceSelectedVoices.length;
      }
      return !this.$store.selectedBatchModels.length;
    },
    batchSubmitLabel() {
      if (this.$store.batchProgress) return 'Generating...';
      if (!this.$store.batchMode) return 'Generate';
      if (this.$store.batchTab === 'multivoice') {
        const n = this.$store.multivoiceSelectedVoices.length;
        return 'Generate with ' + n + ' voice' + (n === 1 ? '' : 's');
      }
      return 'Batch Generate';
    },
    hasAvailSttModel() {
      return (this.$store.e2eModels || []).some(m => m.available);
    },
  },
  methods: {
    engineLabel(engine) {
      return engine.charAt(0).toUpperCase() + engine.slice(1);
    },
    capsLabel(m) {
      if (!m.capabilities || !m.capabilities.length) return '';
      return '  [' + m.capabilities.join(', ') + ']';
    },
    unavailLabel(m) {
      return !m.available ? ' [unavailable]' : '';
    },
    onModelChange() {
      // Reset voice fields when model changes
      this.$store.form.speaker_name = '';
      this.$store.form.voice_description = '';
      this.$store.form.sample_voice_file = '';
    },
    async onSubmit() {
      const m = this.$store.form.model;
      const t = this.$store.form.text.trim();
      if (!m || !t) {
        this.genStatus = 'Please select a model and enter text.';
        this.genStatusClass = 'error';
        return;
      }
      const caps = this.capabilities;
      const body = { model: m, input: t, speed: parseFloat(this.$store.form.speed) };

      if (caps.includes('speaker') || caps.includes('voice_blend')) {
        const v = this.$store.form.speaker_name.trim();
        if (v) body.voice = v;
      } else if (caps.includes('voice_prompt')) {
        const v = this.$store.form.voice_description.trim();
        if (!v) { this.genStatus = 'Voice description is required.'; this.genStatusClass = 'error'; return; }
        body.voice = v;
      } else if (caps.includes('voice_clone')) {
        const v = this.$store.form.sample_voice_file.trim();
        if (!v) { this.genStatus = 'A sample voice file is required.'; this.genStatusClass = 'error'; return; }
        body.voice = v;
      }

      body.temperature = parseFloat(this.$store.form.temperature);
      if (this.$store.form.seed) body.seed = parseInt(this.$store.form.seed, 10);
      body.add_pauses = this.$store.form.add_pauses;

      this.$store.loading = true;
      this.genStatus = '';
      this.genStatusClass = '';

      try {
        const resp = await fetch('/v1/audio/speech', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json', 'X-Save-Output': 'true' },
          body: JSON.stringify(body),
        });
        if (!resp.ok) {
          const err = await resp.json().catch(() => ({ detail: 'HTTP ' + resp.status }));
          this.genStatus = err.detail || 'Error ' + resp.status;
          this.genStatusClass = 'error';
          return;
        }

        const data = await fetch('/outputs/detail').then(r => r.json());
        this.$store.outputs = data;

        if (this.$store.e2eEnabled) {
          this.genStatus = 'Transcribing...';
          this.genStatusClass = 'info';
          const sttModel = this.$store.e2eModel;
          const e2eLang = this.$store.e2eLanguage || undefined;
          for (const out of this.$store.outputs) {
            if (out.params && out.params.e2e) continue;
            const origText = out.params && out.params.input;
            if (!origText) continue;
            try {
              const transResult = await this._transcribeOne(out.url, sttModel, e2eLang);
              const similarity = this._wer(origText, transResult.text);
              const e2eData = {
                e2e: {
                  stt_model: sttModel,
                  transcription: transResult.text,
                  detected_language: transResult.detected_language || '',
                  wer: similarity.wer,
                  similarity: similarity.similarity,
                },
              };
              fetch('/output/' + encodeURIComponent(out.name) + '/meta', {
                method: 'PUT',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(e2eData),
              }).catch(() => {});
              if (out.params) Object.assign(out.params, e2eData);
            } catch (_) {}
          }
        }

        this.btnState = 'btn-success';
        this.genStatus = 'Done!';
        this.genStatusClass = 'success';
        setTimeout(() => { this.btnState = ''; }, 1200);
      } catch (err) {
        this.btnState = 'btn-error';
        this.genStatus = 'Network error: ' + err.message;
        this.genStatusClass = 'error';
        setTimeout(() => { this.btnState = ''; }, 600);
      } finally {
        this.$store.loading = false;
      }
    },
    copyCurl() {
      if (!this.curlCommand) return;
      navigator.clipboard.writeText(this.curlCommand).then(() => {
        // Flash feedback
      }).catch(() => {
        const ta = document.createElement('textarea');
        ta.value = this.curlCommand;
        document.body.appendChild(ta);
        ta.select();
        document.execCommand('copy');
        document.body.removeChild(ta);
      });
    },

    // Batch mode methods
    async batchSubmit() {
      const text = this.$store.form.text.trim();
      if (!text) {
        this.genStatus = 'Please enter text.';
        this.genStatusClass = 'error';
        return;
      }

      const tab = this.$store.batchTab;
      const batchId = Date.now().toString(36) + Math.random().toString(36).slice(2, 8);

      let items = [];
      if (tab === 'multivoice') {
        items = this.$store.multivoiceSelectedVoices.map(v => ({
          id: v,
          label: v,
          modelId: this.$store.multivoiceModel,
          voice: v,
        }));
      } else {
        const models = this.$store.selectedBatchModels;
        if (!models.length) {
          this.genStatus = 'Please select at least one model.';
          this.genStatusClass = 'error';
          return;
        }
        const voiceField = tab === 'clone' ? this.$store.form.sample_voice_file.trim() : this.$store.form.voice_description.trim();
        if (!voiceField) {
          this.genStatus = tab === 'clone' ? 'Please select a voice file.' : 'Please enter a voice description.';
          this.genStatusClass = 'error';
          return;
        }
        items = models.map(id => {
          const m = this.$store.models.find(x => x.id === id);
          return { id, label: m ? m.name : id, modelId: id, voice: voiceField };
        });
      }

      const total = items.length;
      this.$store.loading = true;
      this.$store.batchProgress = { current: 0, total, modelName: '' };
      this.$store.batchAbort = false;
      this.$store.batchSummary = null;

      let succeeded = 0;
      let failed = 0;
      const details = [];

      for (let i = 0; i < total; i++) {
        if (this.$store.batchAbort) {
          details.push('Cancelled after ' + i + ' of ' + total);
          break;
        }
        const item = items[i];
        this.$store.batchProgress = { current: i + 1, total, modelName: item.label };

        const body = {
          model: item.modelId,
          input: text,
          voice: item.voice,
          speed: parseFloat(this.$store.form.speed),
          temperature: parseFloat(this.$store.form.temperature),
          add_pauses: this.$store.form.add_pauses,
        };
        if (this.$store.form.seed) body.seed = parseInt(this.$store.form.seed, 10);

        try {
          const resp = await fetch('/v1/audio/speech', {
            method: 'POST',
            headers: {
              'Content-Type': 'application/json',
              'X-Save-Output': 'true',
              'X-Batch-Id': batchId,
              'X-Batch-Seq': String(i),
            },
            body: JSON.stringify(body),
          });
          if (resp.ok) {
            succeeded++;
            details.push(item.label + ': done');
          } else {
            const err = await resp.json().catch(() => ({ detail: 'HTTP ' + resp.status }));
            failed++;
            details.push(item.label + ': ' + (err.detail || 'error'));
          }
        } catch (err) {
          failed++;
          details.push(item.label + ': network error');
        }

        if (i < total - 1 && !this.$store.batchAbort) {
          await new Promise(r => setTimeout(r, this.$store.batchGapMs));
        }
      }

      this.$store.batchProgress = null;
      this.genStatus = 'Finalizing...';
      this.genStatusClass = 'info';

      try {
        const data = await fetch('/outputs/detail').then(r => r.json());
        this.$store.outputs = data;
      } catch {}

      // E2E validation: transcribe all outputs at the end
      let e2eDone = 0;
      let e2eFailed = 0;
      if (this.$store.e2eEnabled && !this.$store.batchAbort && succeeded > 0) {
        const sttModel = this.$store.e2eModel;
        const e2eLang = this.$store.e2eLanguage || undefined;
        this.genStatus = 'Transcribing ' + succeeded + ' output' + (succeeded > 1 ? 's' : '') + '...';
        this.genStatusClass = 'info';

        for (const out of this.$store.outputs) {
          if (out.params && out.params.e2e) continue; // already has results
          const origText = out.params && out.params.input;
          if (!origText) continue;
          try {
            const transResult = await this._transcribeOne(out.url, sttModel, e2eLang);
            const similarity = this._wer(origText, transResult.text);
            const e2eData = {
              e2e: {
                stt_model: sttModel,
                transcription: transResult.text,
                detected_language: transResult.detected_language || '',
                wer: similarity.wer,
                similarity: similarity.similarity,
              },
            };
            fetch('/output/' + encodeURIComponent(out.name) + '/meta', {
              method: 'PUT',
              headers: { 'Content-Type': 'application/json' },
              body: JSON.stringify(e2eData),
            }).catch(() => {});
            if (out.params) Object.assign(out.params, e2eData);
            e2eDone++;
          } catch (err) {
            e2eFailed++;
          }
        }
      }

      this.$store.loading = false;

      const e2eParts = [];
      if (e2eDone) e2eParts.push(e2eDone + ' validated');
      if (e2eFailed) e2eParts.push(e2eFailed + ' stt errors');

      if (!this.$store.batchAbort && failed === 0 && e2eFailed === 0) {
        this.btnState = 'btn-success';
        setTimeout(() => { this.btnState = ''; }, 1200);
      } else if (!this.$store.batchAbort && (failed > 0 || e2eFailed > 0)) {
        this.btnState = 'btn-error';
        setTimeout(() => { this.btnState = ''; }, 600);
      }

      if (this.$store.batchAbort) {
        this.$store.batchSummary = { type: 'warning', message: 'Cancelled. ' + succeeded + ' generated, ' + failed + ' failed.' + (e2eParts.length ? ' ' + e2eParts.join(', ') : ''), detail: details.join('\n') };
      } else {
        const totalDone = succeeded + failed;
        const msg = 'Generated ' + succeeded + ' of ' + totalDone + (totalDone === 1 ? ' item' : ' items');
        const type = failed === 0 && e2eFailed === 0 ? 'success' : 'warning';
        const extra = [];
        if (failed) extra.push(failed + ' failed');
        if (e2eFailed) extra.push(e2eFailed + ' stt errors');
        this.$store.batchSummary = {
          type,
          message: msg + (extra.length ? ' (' + extra.join(', ') + ')' : '') + (e2eParts.length ? ' \u00B7 ' + e2eParts.join(', ') : ''),
          detail: details.join('\n'),
        };
      }

      setTimeout(() => { this.$store.batchSummary = null; }, 15000);
    },

    cancelBatch() {
      this.$store.batchAbort = true;
    },

    toggleBatchModel(id) {
      const idx = this.$store.selectedBatchModels.indexOf(id);
      if (idx >= 0) {
        this.$store.selectedBatchModels.splice(idx, 1);
      } else {
        this.$store.selectedBatchModels.push(id);
      }
    },

    selectAllVisible() {
      const tab = this.$store.batchTab;
      const models = tab === 'clone' ? this.cloneModels : this.promptModels;
      this.$store.selectedBatchModels.splice(0, this.$store.selectedBatchModels.length, ...models.map(m => m.id));
    },

    clearSelected() {
      this.$store.selectedBatchModels = [];
    },

    switchBatchTab(tab) {
      this.$store.batchTab = tab;
      this.$store.selectedBatchModels = [];
      this.$store.multivoiceModel = '';
      this.$store.multivoiceSelectedVoices = [];
    },

    toggleMultivoiceVoice(name) {
      const idx = this.$store.multivoiceSelectedVoices.indexOf(name);
      if (idx >= 0) {
        this.$store.multivoiceSelectedVoices.splice(idx, 1);
      } else {
        this.$store.multivoiceSelectedVoices.push(name);
      }
    },

    selectAllVoices() {
      this.$store.multivoiceSelectedVoices.splice(0, this.$store.multivoiceSelectedVoices.length, ...this.multivoiceVoices);
    },

    async _transcribeOne(audioUrl, sttModel, language) {
      const resp = await fetch(audioUrl);
      const blob = await resp.blob();
      const fd = new FormData();
      fd.append('file', blob, 'audio.wav');
      fd.append('model', sttModel);
      if (language) fd.append('language', language);
      fd.append('response_format', 'json');
      const sttResp = await fetch('/v1/audio/transcriptions', { method: 'POST', body: fd });
      if (!sttResp.ok) {
        const err = await sttResp.json().catch(() => ({ detail: 'HTTP ' + sttResp.status }));
        throw new Error(err.detail || 'Transcription failed');
      }
      return await sttResp.json();
    },

    _wer(original, transcribed) {
      const norm = s => s.toLowerCase().replace(/[^\w\s']/g, '').replace(/\s+/g, ' ').trim();
      const origWords = norm(original).split(/\s+/).filter(Boolean);
      const transWords = norm(transcribed).split(/\s+/).filter(Boolean);
      if (!origWords.length) return { wer: 0, similarity: transWords.length ? 0 : 1 };

      const m = origWords.length;
      const n = transWords.length;
      const dp = Array.from({ length: m + 1 }, () => Array(n + 1).fill(0));
      for (let i = 0; i <= m; i++) dp[i][0] = i;
      for (let j = 0; j <= n; j++) dp[0][j] = j;
      for (let i = 1; i <= m; i++) {
        for (let j = 1; j <= n; j++) {
          const cost = origWords[i - 1] === transWords[j - 1] ? 0 : 1;
          dp[i][j] = Math.min(dp[i - 1][j] + 1, dp[i][j - 1] + 1, dp[i - 1][j - 1] + cost);
        }
      }
      const wer = dp[m][n] / Math.max(m, n, 1);
      const similarity = Math.max(0, 1 - wer);
      return { wer, similarity };
    },
  },
};

// ── Curl Console ─────────────────────────────────────────────────────────────
comps['curl-console'] = {
  template: '#curl-console-template',
  computed: {
    curlText() {
      const store = this.$store;
      const m = store.form.model;
      const t = store.form.text.trim();
      if (!m || !t) return '# Fill out the form to generate a curl command...';

      const model = store.models.find(x => x.id === m);
      const caps = model ? (model.capabilities || []) : [];
      const body = { model: m, input: t, speed: parseFloat(store.form.speed) };

      if (caps.includes('speaker') || caps.includes('voice_blend')) {
        const v = store.form.speaker_name.trim();
        if (v) body.voice = v;
      } else if (caps.includes('voice_prompt')) {
        const v = store.form.voice_description.trim();
        if (v) body.voice = v;
      } else if (caps.includes('voice_clone')) {
        const v = store.form.sample_voice_file.trim();
        if (v) body.voice = v;
      }

      body.temperature = parseFloat(store.form.temperature);
      if (store.form.seed) body.seed = parseInt(store.form.seed, 10);
      body.add_pauses = store.form.add_pauses;

      const json = JSON.stringify(body, null, 2);
      return [
        'curl -X POST http://localhost:8000/v1/audio/speech \\',
        '  -H "Content-Type: application/json" \\',
        '  -H "X-Save-Output: true" \\',
        "  -d '" + json.replace(/'/g, "'\\''") + "' \\",
        '  --output speech.mp3',
      ].join('\n');
    },
    curlReady() {
      return !this.curlText.startsWith('#');
    },
    curlStyle() {
      if (this.curlReady) {
        return { borderColor: 'rgba(63,185,80,0.3)', background: 'rgba(63,185,80,0.12)', color: 'var(--success)' };
      }
      return {};
    },
  },
  methods: {
    copyCurl() {
      if (!this.curlReady) return;
      navigator.clipboard.writeText(this.curlText).then(() => {
        this.$el.querySelector('.curl-copy-btn').textContent = 'Copied!';
        setTimeout(() => { const el = this.$el && this.$el.querySelector('.curl-copy-btn'); if (el) el.textContent = 'Copy'; }, 2000);
      }).catch(() => {
        const ta = document.createElement('textarea');
        ta.value = this.curlText;
        document.body.appendChild(ta);
        ta.select();
        document.execCommand('copy');
        document.body.removeChild(ta);
      });
    },
  },
};

// ── Add Voice Modal ──────────────────────────────────────────────────────────
comps['add-voice-modal'] = {
  template: '#add-voice-modal-template',
  data() {
    return {
      activeTab: 'upload',
      uploadFile: null,
      uploadName: '',
      uploadStatus: '',
      uploadStatusClass: '',
      recordName: '',
      recordStatus: '',
      recordStatusClass: '',
      recording: false,
      _recorder: null,
      _chunks: [],
      _ctx: null,
      _analyser: null,
      _animFrame: null,
      _stream: null,
      stagedFile: null,
      stageTranscription: '',
      stageTranscribing: false,
      stageSaving: false,
      playing: false,
      currentTime: 0,
      duration: 0,
      seeker: false,
      _audio: null,
    };
  },
  computed: {
    hasAvailableStt() {
      return (this.$store.e2eModels || []).some(m => m.available);
    },
  },
  watch: {
    stagedFile(v) { if (!v && this._audio) this.stopPlayer(); },
  },
  methods: {
    switchTab(tab) { this.activeTab = tab; },
    async uploadSubmit() {
      const file = this.uploadFile;
      if (!file) {
        this.uploadStatus = 'Please select a file.';
        this.uploadStatusClass = 'error';
        return;
      }
      const btn = this.$el.querySelector('#upload-btn');
      if (btn) { btn.disabled = true; btn.textContent = 'Uploading...'; }
      try {
        const fd = new FormData();
        fd.append('file', file);
        if (this.uploadName.trim()) fd.append('name', this.uploadName.trim());
        const resp = await fetch('/voice/stage', { method: 'POST', body: fd });
        const data = await resp.json();
        if (!resp.ok) {
          this.uploadStatus = data.detail || 'Error ' + resp.status;
          this.uploadStatusClass = 'error';
          return;
        }
        this.stagedFile = data;
        this.uploadStatus = '';
        this.uploadFile = null;
        this.uploadName = '';
      } catch (err) {
        this.uploadStatus = 'Network error: ' + err.message;
        this.uploadStatusClass = 'error';
      } finally {
        if (btn) { btn.disabled = false; btn.textContent = 'Upload'; }
      }
    },
    generateVoiceName() {
      const adj = ['Velvet','Crimson','Amber','Azure','Cosmic','Neon','Shadow','Crystal','Thunder','Silver','Golden','Midnight','Solar','Lunar','Ember','Frost','Storm','Echo','Phantom','Dusk','Dawn','Flux','Prism','Vortex','Nova','Sonic','Aurora','Iris','Haze','Drift','Pulse','Bloom','Ember','Cinder','Onyx','Jade','Copper','Woven','Quiet','Fading','Bright','Deep'];
      const noun = ['Voice','Echo','Wave','Tone','Chord','Rhythm','Melody','Harmony','Timbre','Resonance','Whisper','Song','Call','Hum','Drift','Pulse','Flow','Stream','Wind','Thunder','Rain','Storm','Light','Dream','Spirit','Bloom','Flux','Sway','Glide','Haze','Ember','Petal','Veil','Hymn','Reed','Chord'];
      return adj[Math.floor(Math.random() * adj.length)] + '_' + noun[Math.floor(Math.random() * noun.length)];
    },
    async startRecording() {
      if (!this.recordName.trim()) {
        this.recordName = this.generateVoiceName();
      }
      if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
        this.recordStatus = window.isSecureContext === false
          ? 'Recording requires HTTPS / localhost. Use http://localhost:8000.'
          : 'Recording not supported in this browser.';
        this.recordStatusClass = 'error';
        return;
      }
      try {
        const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
        this._stream = stream;
        this._chunks = [];
        this._recorder = new MediaRecorder(stream, {
          mimeType: MediaRecorder.isTypeSupported('audio/webm;codecs=opus') ? 'audio/webm;codecs=opus' : 'audio/webm',
        });
        this._recorder.ondataavailable = (e => { if (e.data.size > 0) this._chunks.push(e.data); });
        this._recorder.onstop = () => this.onRecordingStop();
        this._recorder.start();
        this.recording = true;
        this.recordStatus = 'Recording... Speak now.';
        this.recordStatusClass = 'success';

        // Visualizer
        this._ctx = new AudioContext();
        const src = this._ctx.createMediaStreamSource(stream);
        this._analyser = this._ctx.createAnalyser();
        this._analyser.fftSize = 128;
        src.connect(this._analyser);
        this.drawWave();
      } catch (err) {
        this.recordStatus = 'Microphone access denied: ' + err.message;
        this.recordStatusClass = 'error';
      }
    },
    stopRecording() {
      if (this._recorder && this._recorder.state !== 'inactive') this._recorder.stop();
      this.recording = false;
    },
    async onRecordingStop() {
      if (this._stream) { this._stream.getTracks().forEach(t => t.stop()); this._stream = null; }
      if (this._animFrame) { cancelAnimationFrame(this._animFrame); this._animFrame = null; }
      this._analyser = null;
      if (this._ctx) { this._ctx.close(); this._ctx = null; }
      this.recording = false;
      if (!this._chunks.length) return;

      const blob = new Blob(this._chunks, { type: this._recorder ? this._recorder.mimeType : 'audio/webm' });
      const fd = new FormData();
      fd.append('file', blob, this.recordName.trim() + '.webm');
      fd.append('name', this.recordName.trim());
      this.recordStatus = 'Uploading...';
      this.recordStatusClass = '';
      try {
        const resp = await fetch('/voice/stage', { method: 'POST', body: fd });
        const data = await resp.json();
        if (!resp.ok) {
          this.recordStatus = data.detail || 'Error ' + resp.status;
          this.recordStatusClass = 'error';
          return;
        }
        this.stagedFile = data;
        this.recordStatus = '';
        this.recordName = '';
      } catch (err) {
        this.recordStatus = 'Network error: ' + err.message;
        this.recordStatusClass = 'error';
      }
    },
    async transcribeStage() {
      const sttModel = (this.$store.e2eModels || []).find(m => m.available && m.id);
      if (!sttModel) {
        this.stageTranscription = 'No STT model available. Install a Whisper model first.';
        return;
      }
      this.stageTranscribing = true;
      this.stageTranscription = '';
      try {
        const resp = await fetch(this.stagedFile.url);
        const blob = await resp.blob();
        const fd = new FormData();
        fd.append('file', blob, this.stagedFile.name);
        fd.append('model', sttModel.id);
        fd.append('response_format', 'json');
        const sttResp = await fetch('/v1/audio/transcriptions', { method: 'POST', body: fd });
        const d = await sttResp.json();
        if (!sttResp.ok) {
          this.stageTranscription = d.detail || 'Transcription failed';
        } else {
          this.stageTranscription = d.text;
        }
      } catch (err) {
        this.stageTranscription = 'Error: ' + err.message;
      } finally {
        this.stageTranscribing = false;
      }
    },
    async saveStage() {
      this.stageSaving = true;
      try {
        const resp = await fetch('/voice/stage/' + encodeURIComponent(this.stagedFile.name) + '/save', { method: 'POST' });
        const data = await resp.json();
        if (!resp.ok) {
          this.stageSaving = false;
          return;
        }
        this.$store.voiceDetails.unshift(data);
        fetch('/v1/voices').then(r => r.json()).then(d => {
          if (d && d.data) applyVoiceData(this.$store, d.data);
        }).catch(() => {});
        const sel = this.$store.form.model;
        if (sel) {
          const m = this.$store.models.find(mdl => mdl.id === sel);
          if (m && m.voices && m.voices.cloneable && !m.voices.cloneable.includes(data.name)) {
            m.voices.cloneable.push(data.name);
          }
        }
        this.close();
      } catch (err) {
        this.stageSaving = false;
      }
    },
    async discardStage() {
      if (this.stagedFile) {
        try { await fetch('/voice/stage/' + encodeURIComponent(this.stagedFile.name), { method: 'DELETE' }); } catch {}
      }
      this.stagedFile = null;
      this.stageTranscription = '';
      this.activeTab = 'upload';
    },
    drawWave() {
      if (!this._analyser) return;
      const canvas = this.$el.querySelector('#recording-wave');
      if (!canvas) return;
      const ctx = canvas.getContext('2d');
      const bufferLength = this._analyser.frequencyBinCount;
      const dataArray = new Uint8Array(bufferLength);
      const w = canvas.width, h = canvas.height;

      const draw = () => {
        if (!this._analyser) return;
        this._animFrame = requestAnimationFrame(draw);
        this._analyser.getByteTimeDomainData(dataArray);
        ctx.fillStyle = '#0d1117';
        ctx.fillRect(0, 0, w, h);
        ctx.lineWidth = 2;
        ctx.strokeStyle = '#58a6ff';
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
      };
      draw();
    },
    formatTime(s) {
      if (!s || !isFinite(s)) return '0:00';
      const m = Math.floor(s / 60);
      const sec = Math.floor(s % 60);
      return m + ':' + (sec < 10 ? '0' : '') + sec;
    },
    togglePlayer() {
      if (this._audio && this.playing) {
        this.stopPlayer();
        return;
      }
      if (this.$store.currentAudio) {
        this.$store.currentAudio.pause();
        this.$store.currentAudio = null;
        this.$store.currentName = '';
      }
      const audio = new Audio(this.stagedFile.url);
      this._audio = audio;
      this.seeker = true;
      this.playing = true;
      this.$store.currentAudio = audio;
      this.$store.currentName = this.stagedFile.name;

      audio.addEventListener('timeupdate', () => {
        this.currentTime = audio.currentTime;
        this.duration = audio.duration || 0;
      });
      audio.addEventListener('ended', () => this.stopPlayer());
      audio.play().catch(() => this.stopPlayer());
    },
    stopPlayer() {
      if (this._audio) {
        this._audio.pause();
        this._audio = null;
      }
      this.playing = false;
      this.seeker = false;
      this.currentTime = 0;
      this.duration = 0;
      if (this.$store.currentName === this.stagedFile?.name) {
        this.$store.currentAudio = null;
        this.$store.currentName = '';
      }
    },
    seekPlayer(e) {
      const rect = e.currentTarget.getBoundingClientRect();
      const pct = (e.clientX - rect.left) / rect.width;
      const audio = this._audio || this.$store.currentAudio;
      if (audio && audio.duration) audio.currentTime = pct * audio.duration;
    },
    close() {
      this.stopPlayer();
      this.$store.showAddVoice = false;
    },
  },
};

// ── Params Modal ─────────────────────────────────────────────────────────────
comps['params-modal'] = {
  template: '#params-modal-template',
  computed: {
    out() { return this.$store.modalOut; },
  },
  methods: {
    fillForm() {
      const p = this.$store.regenParams;
      if (!p) return;
      if (p.model) {
        const opt = this.$store.models.find(m => m.id === p.model);
        if (opt) {
          this.$store.form.model = p.model;
        }
      }
      if (p.input) this.$store.form.text = p.input;
      if (p.speed != null) {
        this.$store.form.speed = Math.min(4, Math.max(0.25, parseFloat(p.speed)));
      }
      this.$store.form.seed = p.seed != null ? p.seed : null;
      if (p.voice) {
        const model = this.$store.models.find(m => m.id === (p.model || this.$store.form.model));
        const caps = model ? (model.capabilities || []) : [];
        if (caps.includes('speaker') || caps.includes('voice_blend')) this.$store.form.speaker_name = p.voice;
        else if (caps.includes('voice_prompt')) this.$store.form.voice_description = p.voice;
        else if (caps.includes('voice_clone')) this.$store.form.sample_voice_file = p.voice;
      }
      this.close();
    },
    deleteOut() {
      const out = this.out;
      if (!out) return;
      if (!confirm('Delete "' + out.name + '"?')) return;
      fetch('/output/' + encodeURIComponent(out.name), { method: 'DELETE' }).then(r => {
        if (!r.ok) return;
        if (this.$store.currentName === out.name) { if (this.$store.currentAudio) this.$store.currentAudio.pause(); this.$store.currentAudio = null; this.$store.currentName = ''; }
        this.$store.outputs = this.$store.outputs.filter(o => o.name !== out.name);
        this.close();
      });
    },
    close() { this.$store.showParams = false; },
    e2eResultClass(similarity) {
      if (similarity >= 0.9) return 'e2e-good';
      if (similarity >= 0.7) return 'e2e-ok';
      return 'e2e-bad';
    },
  },
};

// ── Install Modal ────────────────────────────────────────────────────────────
comps['install-modal'] = {
  template: '#install-modal-template',
  computed: {
    groups() {
      const g = {};
      this.$store.models.forEach(m => {
        if (!m.install) return;
        const e = m.engine || 'other';
        if (!g[e]) g[e] = [];
        g[e].push(m);
      });
      return g;
    },
    engineNames() {
      return Object.keys(this.groups).sort();
    },
  },
  methods: {
    copyCmd(cmd) {
      navigator.clipboard.writeText(cmd).then(() => {
        // Flash feedback handled in template
      }).catch(() => {
        const ta = document.createElement('textarea');
        ta.value = cmd;
        document.body.appendChild(ta);
        ta.select();
        document.execCommand('copy');
        document.body.removeChild(ta);
      });
    },
    close() { this.$store.showInstall = false; },
  },
};

// ── Batch Detail Modal ────────────────────────────────────────────────────
comps['batch-detail-modal'] = {
  template: '#batch-detail-modal-template',
  computed: {
    items() { return this.$store.batchDetailItems; },
    batchType() {
      if (!this.items.length) return 'unknown';
      const models = new Set();
      const voices = new Set();
      this.items.forEach(i => {
        if (i.params) {
          models.add(i.params.model);
          if (i.params.voice) voices.add(i.params.voice);
        }
      });
      if (voices.size === 1) {
        const v = [...voices][0];
        return v.endsWith('.wav') ? 'clone' : 'prompt';
      }
      return 'multivoice';
    },
    modalTitle() {
      if (!this.items.length) return 'Batch';
      const map = { multivoice: 'Speaker', clone: 'Voice Clone', prompt: 'Voice Prompt' };
      return 'Batch - ' + (map[this.batchType] || '');
    },
    sharedText() {
      return (this.items[0] && this.items[0].params && this.items[0].params.input) || '';
    },
    singularInfo() {
      if (!this.items.length) return null;
      if (this.batchType === 'multivoice') {
        const mId = this.items[0].params && this.items[0].params.model;
        if (!mId) return null;
        const m = this.$store.models.find(x => x.id === mId);
        return { label: 'Model', value: m ? m.name : mId };
      }
      const v = this.items[0].params && this.items[0].params.voice;
      if (!v) return null;
      const label = this.batchType === 'clone' ? 'Voice' : 'Voice Prompt';
      return { label, value: this.batchType === 'clone' ? v : '"' + v + '"' };
    },
    totalGenerated() {
      return this.items.length + ' generated';
    },
  },
  methods: {
    itemDiff(item) {
      const p = item.params || {};
      if (this.batchType === 'multivoice') return p.voice || '';
      const m = this.$store.models.find(x => x.id === p.model);
      return m ? m.name : (p.model || '');
    },
    e2eResultClass(similarity) {
      if (similarity >= 0.9) return 'e2e-good';
      if (similarity >= 0.7) return 'e2e-ok';
      return 'e2e-bad';
    },
    close() {
      this.$store.showBatchDetail = false;
      this.$store.batchDetailItems = [];
      this.$store.batchDetailLabel = '';
    },
    deleteItem(name) {
      if (!confirm('Delete "' + name + '"?')) return;
      fetch('/output/' + encodeURIComponent(name), { method: 'DELETE' }).then(r => {
        if (!r.ok) return;
        if (this.$store.currentName === name) { this.$store.currentAudio.pause(); this.$store.currentAudio = null; this.$store.currentName = ''; }
        this.$store.batchDetailItems = this.$store.batchDetailItems.filter(o => o.name !== name);
        this.$store.outputs = this.$store.outputs.filter(o => o.name !== name);
      });
    },
  },
};

})();

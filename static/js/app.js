const { createApp } = Vue;

const app = createApp({
  mounted() {
    this.loadFast();
    this.loadOutputs();
  },
  methods: {
    async loadFast() {
      try {
        const [modelData, voiceData, sttData] = await Promise.all([
          fetch('/v1/models?extras=true').then(r => r.json()),
          fetch('/v1/voices').then(r => r.json()),
          fetch('/v1/stt/models').then(r => r.json()).catch(() => ({ data: [] })),
        ]);
        this.$store.models = modelData.data || (Array.isArray(modelData) ? modelData : []);
        applyVoiceData(this.$store, voiceData.data || []);
        this.$store.e2eModels = sttData.data || [];
        const avail = (this.$store.e2eModels || []).filter(m => m.available);
        if (avail.length) this.$store.e2eModel = avail[0].id;
      } catch {
        console.warn('Sonus: failed to load models/voices');
      }
    },
    async loadOutputs() {
      this.$store.outputsLoading = true;
      try {
        const outData = await fetch('/outputs/detail').then(r => r.json());
        this.$store.outputs = outData;
      } catch {
        console.warn('Sonus: failed to load outputs');
      } finally {
        this.$store.outputsLoading = false;
      }
    },
  },
});

app.config.globalProperties.$store = store;

Object.entries(window.__comps || {}).forEach(([name, def]) => {
  app.component(name, def);
});

app.mount('#app');

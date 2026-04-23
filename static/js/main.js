const socket = io();
let trainingTotal = 0;
let trainingDone = 0;
let features = [];

// Socket events
socket.on('training_progress', (data) => {
  document.getElementById('cur-layer').textContent = data.layer;
  document.getElementById('cur-epoch').textContent = `${data.epoch}/${data.total_epochs}`;
  document.getElementById('cur-loss').textContent = data.loss.toFixed(6);
  const pct = (data.epoch / data.total_epochs) * 100;
  document.getElementById('progress-bar').style.width = pct + '%';
  document.getElementById('train-status').textContent = `Pre-training Layer ${data.layer}`;
  document.getElementById('train-phase').textContent = 'PRE-TRAIN';
});

socket.on('training_status', (data) => {
  document.getElementById('train-status').textContent = data.message;
  document.getElementById('train-phase').textContent = data.phase.toUpperCase();
  if (data.phase === 'finetune') {
    document.getElementById('progress-bar').style.width = '80%';
  }
});

socket.on('training_done', async (data) => {
  document.getElementById('progress-bar').style.width = '100%';
  document.getElementById('train-status').textContent = '✅ Training Complete!';
  updatePill('pill-model', true);
  showToast('DBN training complete!', 'success');
  await loadChart('training');
  await updateStatus();
});

socket.on('training_error', (data) => {
  showToast('Training error: ' + data.message, 'error');
  document.getElementById('training-panel').classList.add('hidden');
});

// Main functions
async function loadData() {
  showToast('Loading UCI datasets...', 'info');
  try {
    const res = await fetch('/api/load_data', { method: 'POST' });
    const data = await res.json();
    if (data.status === 'success') {
      features = data.stats.features;
      updatePill('pill-data', true);
      const box = document.getElementById('data-stats');
      box.innerHTML = `
        <b>Shape:</b> ${data.stats.shape[0]} × ${data.stats.shape[1]}<br>
        <b>Features:</b> ${data.stats.features.length}<br>
        <b>Missing:</b> ${data.stats.missing}
      `;
      box.classList.remove('hidden');
      populateFeatureSelect(data.stats.features);
      showSampleTable(data.sample, data.stats.features);
      showToast('Dataset loaded successfully!', 'success');
    } else {
      showToast('Error: ' + data.message, 'error');
    }
  } catch (e) {
    showToast('Network error: ' + e.message, 'error');
  }
}

async function trainModel() {
  const panel = document.getElementById('training-panel');
  panel.classList.remove('hidden');
  document.getElementById('progress-bar').style.width = '0%';

  const params = {
    pretrain_epochs: parseInt(document.getElementById('pretrain-epochs').value),
    finetune_epochs: parseInt(document.getElementById('finetune-epochs').value),
    lr: parseFloat(document.getElementById('lr-select').value),
    batch_size: parseInt(document.getElementById('batch-select').value),
    layers: [18, 64, 32, 16]
  };

  try {
    const res = await fetch('/api/train', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify(params)
    });
    const data = await res.json();
    if (data.status !== 'started') {
      showToast('Error: ' + data.message, 'error');
    }
  } catch (e) {
    showToast('Network error: ' + e.message, 'error');
  }
}

async function generateData() {
  showToast('Generating synthetic patient records...', 'info');
  const params = {
    n_samples: parseInt(document.getElementById('n-samples').value),
    temperature: parseFloat(document.getElementById('temperature').value),
    noise: parseFloat(document.getElementById('noise-level').value)
  };

  try {
    const res = await fetch('/api/generate', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify(params)
    });
    const data = await res.json();
    if (data.status === 'success') {
      updatePill('pill-synth', true);
      features = data.columns;
      populateFeatureSelect(data.columns);
      showSampleTable(data.sample, data.columns);
      await loadChart('pca');
      await loadChart('correlation');
      await loadDistChart();
      showToast(`Generated ${data.n_generated} synthetic records!`, 'success');
    } else {
      showToast('Error: ' + data.message, 'error');
    }
  } catch (e) {
    showToast('Error: ' + e.message, 'error');
  }
}

async function evaluatePrivacy() {
  showToast('Running privacy evaluation...', 'info');
  try {
    const res = await fetch('/api/evaluate_privacy', { method: 'POST' });
    const data = await res.json();
    if (data.status === 'success') {
      updatePill('pill-privacy', true);
      updateMetrics(data.results, null);
      await loadChart('privacy_radar');
      showToast('Privacy evaluation complete!', 'success');
    } else {
      showToast('Error: ' + data.message, 'error');
    }
  } catch (e) {
    showToast('Error: ' + e.message, 'error');
  }
}

async function evaluateML() {
  showToast('Running ML evaluation (TSTR vs TRTR)...', 'info');
  try {
    const res = await fetch('/api/evaluate_ml', { method: 'POST' });
    const data = await res.json();
    if (data.status === 'success') {
      updatePill('pill-ml', true);
      updateMetrics(null, data.results);
      await loadChart('ml_comparison');
      showToast('ML evaluation complete!', 'success');
    } else {
      showToast('Error: ' + data.message, 'error');
    }
  } catch (e) {
    showToast('Error: ' + e.message, 'error');
  }
}

async function runLLMAnalysis() {
  showToast('Consulting LLaMA-3.3-70B AI...', 'info');
  document.getElementById('ai-analysis-text').textContent = '⏳ Analyzing with AI...';
  try {
    const res = await fetch('/api/llm_analysis', { method: 'POST' });
    const data = await res.json();
    if (data.status === 'success') {
      document.getElementById('ai-analysis-text').textContent = data.insight;
      document.getElementById('ai-arch-text').textContent = data.architecture;
      showTab('tab-analysis');
      showToast('AI analysis complete!', 'success');
    } else {
      document.getElementById('ai-analysis-text').textContent = 'Error: ' + data.message;
      showToast('Error: ' + data.message, 'error');
    }
  } catch (e) {
    showToast('Error: ' + e.message, 'error');
  }
}

async function getPatientNarrative() {
  const idx = parseInt(document.getElementById('narrative-idx').value) || 0;
  document.getElementById('ai-narrative-text').textContent = '⏳ Generating clinical narrative...';
  try {
    const res = await fetch('/api/patient_narrative', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({ idx })
    });
    const data = await res.json();
    if (data.status === 'success') {
      document.getElementById('ai-narrative-text').textContent = data.narrative;
      const recDiv = document.getElementById('patient-record-display');
      recDiv.innerHTML = Object.entries(data.record).map(([k, v]) =>
        `<div class="rec-field"><span class="rec-key">${k}</span><span class="rec-val">${typeof v === 'number' ? v.toFixed(2) : v}</span></div>`
      ).join('');
      recDiv.classList.remove('hidden');
    } else {
      showToast('Error: ' + data.message, 'error');
    }
  } catch (e) {
    showToast('Error: ' + e.message, 'error');
  }
}

async function exportData() {
  showToast('Preparing export...', 'info');
  try {
    const res = await fetch('/api/export');
    const data = await res.json();
    if (data.status === 'success') {
      const cols = data.columns;
      const rows = data.data;
      let csv = cols.join(',') + '\n';
      rows.forEach(r => { csv += cols.map(c => r[c]).join(',') + '\n'; });
      const blob = new Blob([csv], { type: 'text/csv' });
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url; a.download = 'synthetic_patient_records.csv';
      a.click();
      showToast('Export complete!', 'success');
    }
  } catch (e) {
    showToast('Error: ' + e.message, 'error');
  }
}

async function loadChart(name) {
  const endpoints = {
    'training': '/api/charts/training',
    'pca': '/api/charts/pca',
    'correlation': '/api/charts/correlation',
    'privacy_radar': '/api/charts/privacy_radar',
    'ml_comparison': '/api/charts/ml_comparison'
  };
  const plotIds = {
    'training': 'plot-training',
    'pca': 'plot-pca',
    'correlation': 'plot-corr',
    'privacy_radar': 'plot-radar',
    'ml_comparison': 'plot-ml'
  };
  if (!endpoints[name]) return;
  try {
    const res = await fetch(endpoints[name]);
    const fig = await res.json();
    if (fig.data) {
      Plotly.newPlot(plotIds[name], fig.data, {
        ...fig.layout,
        paper_bgcolor: 'rgba(0,0,0,0)',
        plot_bgcolor: 'rgba(0,0,0,0)',
        font: { color: '#e2e8f0', size: 11 },
        margin: { t: 40, l: 50, r: 20, b: 40 }
      }, { responsive: true, displayModeBar: false });
    }
  } catch (e) {
    console.warn('Chart load failed:', name, e);
  }
}

async function loadDistChart() {
  const sel = document.getElementById('feature-select');
  const feature = sel.value;
  if (!feature) return;
  try {
    const res = await fetch(`/api/charts/distribution?feature=${encodeURIComponent(feature)}`);
    const fig = await res.json();
    if (fig.data) {
      Plotly.newPlot('plot-dist', fig.data, {
        ...fig.layout,
        paper_bgcolor: 'rgba(0,0,0,0)',
        plot_bgcolor: 'rgba(0,0,0,0)',
        font: { color: '#e2e8f0', size: 11 },
        margin: { t: 40, l: 50, r: 20, b: 40 }
      }, { responsive: true, displayModeBar: false });
    }
  } catch (e) {}
}

function populateFeatureSelect(featureList) {
  const sel = document.getElementById('feature-select');
  sel.innerHTML = featureList.map(f => `<option value="${f}">${f}</option>`).join('');
}

function showSampleTable(records, cols) {
  if (!records || !records.length) return;
  const panel = document.getElementById('table-panel');
  const container = document.getElementById('table-container');
  const allCols = cols || Object.keys(records[0]);
  let html = '<table><thead><tr>' + allCols.map(c => `<th>${c}</th>`).join('') + '</tr></thead><tbody>';
  records.forEach(r => {
    html += '<tr>' + allCols.map(c => `<td>${typeof r[c] === 'number' ? r[c].toFixed(3) : r[c]}</td>`).join('') + '</tr>';
  });
  html += '</tbody></table>';
  container.innerHTML = html;
}

function updateMetrics(privacy, ml) {
  if (privacy) {
    const sf = privacy.statistical_fidelity || {};
    const cp = privacy.correlation_preservation || {};
    const mi = privacy.membership_inference || {};
    document.getElementById('m-privacy').textContent = (privacy.overall_privacy_score || 0).toFixed(1) + '%';
    document.getElementById('m-fidelity').textContent = (sf.statistical_fidelity_score || 0).toFixed(1) + '%';
    document.getElementById('m-corr').textContent = (cp.correlation_preservation_score || 0).toFixed(1) + '%';
    document.getElementById('m-grade').textContent = privacy.grade || '—';
    document.getElementById('m-risk').textContent = mi.risk_level || '—';
  }
  if (ml) {
    document.getElementById('m-utility').textContent = (ml.utility_preservation || 0).toFixed(1) + '%';
  }
}

function updatePill(id, active) {
  const el = document.getElementById(id);
  if (el) el.classList.toggle('active', active);
}

async function updateStatus() {
  try {
    const res = await fetch('/api/status');
    const s = await res.json();
    updatePill('pill-data', s.data_loaded);
    updatePill('pill-model', s.model_trained);
    updatePill('pill-synth', s.synthetic_generated);
    updatePill('pill-privacy', s.privacy_evaluated);
    updatePill('pill-ml', s.ml_evaluated);
  } catch (e) {}
}

function showTab(tabId) {
  document.querySelectorAll('.tab-content').forEach(t => t.classList.remove('active'));
  document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
  document.getElementById(tabId).classList.add('active');
  event.target.classList.add('active');
}

function showToast(msg, type = 'info') {
  const t = document.getElementById('toast');
  t.textContent = msg;
  t.className = `toast ${type} show`;
  setTimeout(() => { t.classList.remove('show'); }, 3500);
}

// Init
updateStatus();

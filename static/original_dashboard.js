'use strict';

const byId = id => document.getElementById(id);
const classLabels = {N: 'Normal', V: 'PVC', S: 'SVE', F: 'Fusion', Q: 'Unclassifiable'};
const classColors = {N: '#10b981', V: '#ef4444', S: '#3b82f6', F: '#8b5cf6', Q: '#6b7280'};
const fileInput = byId('fileInput');
const dropZone = byId('dropZone');
const analyzeButton = byId('analyzeBtn');
let selectedFile = null;
let currentAnalysisData = null;

function status(message, isError = false) {
  const element = byId('uploadStatus');
  element.textContent = message;
  element.className = `mt-3 text-sm ${isError ? 'text-red-600' : 'text-gray-600'}`;
}

function handleFile(file) {
  if (!file) return;
  const extension = file.name.toLowerCase().split('.').pop();
  if (!['csv', 'txt', 'dat'].includes(extension)) {
    selectedFile = null;
    fileInput.value = '';
    byId('fileInfo').classList.add('hidden');
    analyzeButton.disabled = true;
    status('Choose a CSV, TXT, or DAT ECG file.', true);
    return;
  }
  if (file.size > 2 * 1024 * 1024) {
    selectedFile = null;
    fileInput.value = '';
    byId('fileInfo').classList.add('hidden');
    analyzeButton.disabled = true;
    status('The upload must be smaller than 2 MB.', true);
    return;
  }
  selectedFile = file;
  byId('fileName').textContent = file.name;
  byId('fileSize').textContent = `${(file.size / 1024).toFixed(1)} KB`;
  byId('fileInfo').classList.remove('hidden');
  const isDat = extension === 'dat';
  byId('headerField').classList.toggle('hidden', !isDat);
  byId('encodingField').classList.toggle('hidden', !isDat);
  if (!isDat) byId('headerInput').value = '';
  analyzeButton.disabled = false;
  status(isDat ? 'DAT selected. Add its matching .hea file when available.' : 'File selected and ready to analyze.');
}

dropZone.addEventListener('click', event => {
  if (event.target.closest('button')) event.preventDefault();
  fileInput.click();
});
dropZone.addEventListener('dragover', event => {
  event.preventDefault();
  dropZone.classList.add('drag-over');
});
dropZone.addEventListener('dragleave', () => dropZone.classList.remove('drag-over'));
dropZone.addEventListener('drop', event => {
  event.preventDefault();
  dropZone.classList.remove('drag-over');
  handleFile(event.dataTransfer.files[0]);
});
fileInput.addEventListener('change', () => handleFile(fileInput.files[0]));
byId('removeFile').addEventListener('click', () => {
  selectedFile = null;
  fileInput.value = '';
  byId('headerInput').value = '';
  byId('fileInfo').classList.add('hidden');
  byId('headerField').classList.add('hidden');
  byId('encodingField').classList.add('hidden');
  byId('resultsContainer').classList.add('hidden');
  byId('emptyState').classList.remove('hidden');
  analyzeButton.disabled = true;
  status('Choose a recording to begin.');
});

analyzeButton.addEventListener('click', analyzeRecording);
async function analyzeRecording() {
  if (!selectedFile) return;
  analyzeButton.disabled = true;
  byId('emptyState').classList.add('hidden');
  byId('resultsContainer').classList.add('hidden');
  byId('loadingState').classList.remove('hidden');
  status('Analyzing the recording with the saved Random Forest model…');
  const data = new FormData();
  data.append('file', selectedFile);
  data.append('sampling_rate', byId('samplingRate').value || '360');
  data.append('dat_encoding', byId('datEncoding').value);
  const header = byId('headerInput').files[0];
  if (header) data.append('header', header);
  try {
    const response = await fetch('/api/analyze', {
      method: 'POST', body: data,
      headers: {'X-CSRF-Token': document.querySelector('meta[name="csrf-token"]').content}
    });
    const result = await response.json();
    if (!response.ok || !result.success) throw new Error(result.error || 'Analysis could not be completed.');
    currentAnalysisData = result;
    displayResults(result);
    byId('loadingState').classList.add('hidden');
    byId('resultsContainer').classList.remove('hidden');
    status(`Complete. ${result.classified_beats} beats classified with the Random Forest model.`);
    byId('resultsContainer').scrollIntoView({behavior: 'smooth', block: 'start'});
  } catch (error) {
    byId('loadingState').classList.add('hidden');
    byId('emptyState').classList.remove('hidden');
    status(error.message, true);
  } finally {
    analyzeButton.disabled = false;
  }
}

function displayResults(result) {
  const distribution = result.distribution || {};
  const total = result.classified_beats || 0;
  const normal = distribution.N || 0;
  const ectopic = Math.max(0, total - normal);
  byId('totalBeats').textContent = total;
  byId('heartRate').textContent = result.heart_rate_bpm;
  byId('abnormalBeats').textContent = ectopic;
  byId('normalPercent').textContent = `${total ? (normal * 100 / total).toFixed(1) : '0.0'}%`;
  byId('signalQuality').textContent = `${result.sampling_rate} Hz · educational analysis`;

  const ectopicTypes = Object.entries(distribution).filter(([type, count]) => type !== 'N' && count > 0);
  const alertCard = byId('criticalFindingsCard');
  const alertList = byId('criticalFindingsList');
  alertList.replaceChildren();
  alertCard.classList.toggle('hidden', ectopicTypes.length === 0);
  for (const [type, count] of ectopicTypes) {
    const item = document.createElement('div');
    item.className = 'rounded-lg border border-gray-200 bg-gray-50 p-4';
    const title = document.createElement('p');
    title.className = 'font-semibold text-gray-800';
    title.textContent = classLabels[type] || type;
    const value = document.createElement('p');
    value.className = 'mt-1 text-2xl font-bold';
    value.style.color = classColors[type] || '#374151';
    value.textContent = `${count} predicted`;
    item.append(title, value);
    alertList.appendChild(item);
  }

  const hrv = result.rr_variability || {};
  byId('sdnn').textContent = `${hrv.sdnn_ms ?? 0} ms`;
  byId('rmssd').textContent = `${hrv.rmssd_ms ?? 0} ms`;
  byId('pnn50').textContent = `${hrv.pnn50_percent ?? 0}%`;
  byId('sdnnBar').style.width = `${Math.min((hrv.sdnn_ms || 0) / 2, 100)}%`;
  byId('rmssdBar').style.width = `${Math.min((hrv.rmssd_ms || 0) / 2, 100)}%`;
  byId('pnn50Bar').style.width = `${Math.min(hrv.pnn50_percent || 0, 100)}%`;

  const findings = byId('clinicalFindings');
  findings.replaceChildren();
  for (const note of result.notes || []) {
    const row = document.createElement('p');
    row.className = 'rounded-lg bg-gray-50 p-3 text-sm text-gray-700';
    row.textContent = note;
    findings.appendChild(row);
  }

  const beats = result.beats || [];
  const body = byId('beatDetailsBody');
  body.replaceChildren();
  for (const beat of beats) {
    const row = document.createElement('tr');
    row.className = 'border-t border-gray-100';
    for (const text of [beat.index, beat.time_seconds, `${beat.type} · ${beat.label}`, `${(beat.confidence * 100).toFixed(1)}%`]) {
      const cell = document.createElement('td');
      cell.className = 'p-3';
      cell.textContent = text;
      row.appendChild(cell);
    }
    body.appendChild(row);
  }

  const wave = result.waveform || {seconds: [], amplitude: [], beat_seconds: [], beat_amplitude: []};
  const traces = [{x: wave.seconds, y: wave.amplitude, type: 'scatter', mode: 'lines', name: 'ECG signal',
    line: {color: '#2563eb', width: 1.4}, hovertemplate: '%{x:.2f} s<br>%{y:.3f} mV<extra></extra>'}];
  for (const type of Object.keys(classLabels)) {
    const points = beats.map((beat, i) => beat.type === type ?
      {beat, x: wave.beat_seconds[i], y: wave.beat_amplitude[i]} : null).filter(Boolean);
    if (!points.length) continue;
    traces.push({x: points.map(point => point.x), y: points.map(point => point.y), type: 'scatter', mode: 'markers',
      name: classLabels[type], marker: {color: classColors[type], size: 7},
      text: points.map(point => `${classLabels[type]} · ${point.beat.confidence ? (point.beat.confidence * 100).toFixed(1) + '% vote probability' : 'beat'}`),
      hovertemplate: '%{text}<br>%{x:.2f} s<extra></extra>'});
  }
  Plotly.newPlot('ecgPlot', traces, {margin: {t: 16, r: 20, b: 48, l: 54},
    xaxis: {title: 'Time (seconds)', gridcolor: '#e5e7eb'}, yaxis: {title: 'Amplitude (mV)', gridcolor: '#e5e7eb'},
    legend: {orientation: 'h', y: 1.13}, paper_bgcolor: '#fff', plot_bgcolor: '#fff'},
    {responsive: true, displaylogo: false, modeBarButtonsToRemove: ['lasso2d', 'select2d']});
  const types = Object.keys(classLabels).filter(type => distribution[type] > 0);
  Plotly.newPlot('pieChart', [{values: types.map(type => distribution[type]), labels: types.map(type => `${type} · ${classLabels[type]}`),
    type: 'pie', marker: {colors: types.map(type => classColors[type])}, textinfo: 'label+value', hoverinfo: 'label+value+percent'}],
    {margin: {t: 8, r: 8, b: 8, l: 8}, showlegend: false}, {responsive: true, displaylogo: false});
}

byId('viewToggle').addEventListener('click', () => {
  const details = byId('beatDetails');
  const showing = details.classList.toggle('hidden') === false;
  byId('viewToggle').innerHTML = `<i class="fas fa-table mr-2"></i>${showing ? 'Hide beat details' : 'Show beat details'}`;
});
byId('exportJSON').addEventListener('click', () => {
  if (!currentAnalysisData) return;
  download(JSON.stringify(currentAnalysisData, null, 2), 'application/json', 'ecg-analysis.json');
});
byId('exportPDF').addEventListener('click', () => window.print());
function download(content, type, filename) {
  const link = document.createElement('a');
  const url = URL.createObjectURL(new Blob([content], {type}));
  link.href = url;
  link.download = filename;
  link.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

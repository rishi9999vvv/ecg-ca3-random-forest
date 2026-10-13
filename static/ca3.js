'use strict';
const el = id => document.getElementById(id);
const labels = {N:'Normal',V:'PVC',S:'SVE',F:'Fusion',Q:'Unclassifiable'};
const colors = {N:'#087f80',V:'#cf5151',S:'#d49529',F:'#8966ae',Q:'#86969c'};
let result = null;
el('file').addEventListener('change', () => {
  const file=el('file').files[0], isDat=file?.name.toLowerCase().endsWith('.dat');
  el('filename').textContent=file?.name || 'CSV/TXT amplitudes, or a DAT signal with its optional WFDB header.';
  el('header-field').hidden=!isDat;
  el('dat-encoding-field').hidden=!isDat;
  if(!isDat)el('header').value='';
});
el('upload-form').addEventListener('submit', async event => {
  event.preventDefault();const button = el('analyze');button.disabled = true;button.textContent = 'Analyzing recording…';
  el('status').className = '';el('status').textContent = 'Filtering signal and classifying heartbeat windows…';el('results').hidden = true;result = null;
  for (const id of ['heart-rate','detected','classified','duration']) el(id).textContent='—';
  el('recording-tag').textContent = 'Analyzing';el('chart-empty').hidden = false;el('wave-start').disabled = true;draw();
  try {
    const response = await fetch('/api/analyze', {method:'POST',body:new FormData(event.target),headers:{'X-CSRF-Token':document.querySelector('meta[name="csrf-token"]').content}});
    const data = await response.json();if (!response.ok || !data.success) throw new Error(data.error || 'Analysis could not be completed.');
    result = data;render();el('status').textContent = `Complete. ${data.classified_beats} beats classified with the Random Forest model.`;
  } catch(error) {el('status').className='error';el('status').textContent=error.message;el('recording-tag').textContent = 'Upload failed';}
  finally {button.disabled=false;button.textContent='Analyze recording →';}
});
function render() {
  el('heart-rate').textContent=result.heart_rate_bpm;el('detected').textContent=result.detected_beats;el('classified').textContent=result.classified_beats;el('duration').textContent=result.duration_seconds;
  el('recording-tag').textContent=`${result.sampling_rate} Hz · analyzed`;el('sdnn').textContent=`${result.rr_variability.sdnn_ms} ms`;el('rmssd').textContent=`${result.rr_variability.rmssd_ms} ms`;el('pnn50').textContent=`${result.rr_variability.pnn50_percent}%`;
  el('distribution').replaceChildren();
  for (const [type,label] of Object.entries(labels)) {
    const count=result.distribution[type],percent=100*count/result.classified_beats;
    const row=document.createElement('div');row.className='dist-row';const title=document.createElement('span');title.textContent=label;
    const bar=document.createElement('div');bar.className='bar';const fill=document.createElement('div');fill.style.width=`${percent}%`;fill.style.background=colors[type];bar.append(fill);
    const number=document.createElement('span');number.textContent=`${count} (${percent.toFixed(0)}%)`;row.append(title,bar,number);el('distribution').append(row);
  }
  el('beat-table').replaceChildren();
  for (const beat of result.beats) {const row=document.createElement('tr');for (const value of [beat.index,beat.time_seconds,`${beat.type} · ${beat.label}`,`${(beat.confidence*100).toFixed(1)}%`]) {const cell=document.createElement('td');cell.textContent=value;row.append(cell);}el('beat-table').append(row);}
  el('results').hidden=false;el('chart-empty').hidden=true;el('wave-start').max=Math.max(0,Math.ceil(result.duration_seconds-10));el('wave-start').value=0;el('wave-start').disabled=false;draw();
}
function draw() {
  const canvas=el('waveform'),box=canvas.getBoundingClientRect(),scale=window.devicePixelRatio||1;canvas.width=Math.max(1,box.width*scale);canvas.height=Math.max(1,box.height*scale);
  const ctx=canvas.getContext('2d');ctx.scale(scale,scale);const w=box.width,h=box.height,left=48,right=14,top=18,bottom=32;ctx.clearRect(0,0,w,h);
  const start=Number(el('wave-start').value),end=Math.min(start+10,result?.duration_seconds||10);el('window-label').textContent=`${start}–${end} seconds`;
  const wave=result?.waveform,points=wave ? wave.seconds.map((x,i)=>[x,wave.amplitude[i]]).filter(p=>p[0]>=start&&p[0]<=end) : [];
  const low=points.length?Math.min(...points.map(p=>p[1])):-1,high=points.length?Math.max(...points.map(p=>p[1])):1,pad=Math.max((high-low)*.15,.01),min=low-pad,max=high+pad;
  const x=t=>left+(t-start)/(end-start||1)*(w-left-right),y=v=>top+(max-v)/(max-min)*(h-top-bottom);
  ctx.font='10px Segoe UI';ctx.fillStyle='#7d959b';ctx.strokeStyle='#e4eded';ctx.lineWidth=1;
  for(let i=0;i<=5;i++){const tx=left+i/5*(w-left-right);ctx.beginPath();ctx.moveTo(tx,top);ctx.lineTo(tx,h-bottom);ctx.stroke();ctx.fillText((start+i/5*(end-start)).toFixed(1),tx-8,h-12);}
  for(let i=0;i<=4;i++){const ty=top+i/4*(h-top-bottom);ctx.beginPath();ctx.moveTo(left,ty);ctx.lineTo(w-right,ty);ctx.stroke();ctx.fillText((max-i/4*(max-min)).toFixed(2),5,ty+3);}
  if(!points.length)return;ctx.beginPath();ctx.strokeStyle='#087f80';ctx.lineWidth=1.3;points.forEach(([t,v],i)=>i?ctx.lineTo(x(t),y(v)):ctx.moveTo(x(t),y(v)));ctx.stroke();
  result.beats.forEach((beat,i)=>{const t=wave.beat_seconds[i];if(t<start||t>end)return;ctx.beginPath();ctx.fillStyle=colors[beat.type];ctx.arc(x(t),y(wave.beat_amplitude[i]),3,0,2*Math.PI);ctx.fill();});
}
el('wave-start').addEventListener('input',draw);new ResizeObserver(draw).observe(el('waveform'));
function download(content,type,name){const url=URL.createObjectURL(new Blob([content],{type}));const link=document.createElement('a');link.href=url;link.download=name;link.click();setTimeout(()=>URL.revokeObjectURL(url),1000);}
el('download-json').addEventListener('click',()=>{if(result)download(JSON.stringify(result,null,2),'application/json','ecg-analysis.json');});
el('download-csv').addEventListener('click',()=>{if(result)download('beat,time_seconds,class,confidence\n'+result.beats.map(b=>`${b.index},${b.time_seconds},${b.type},${b.confidence}`).join('\n'),'text/csv','ecg-beats.csv');});
el('print').addEventListener('click',()=>window.print());

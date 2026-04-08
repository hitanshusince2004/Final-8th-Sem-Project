const API = '/api';
let mapInstance, tsChart, curveChart, areaChart, lstChart, waterChart;
let spectralChart, elevChart, scatterChart, precipChart, sbLossChart, sbAccChart;
let numPage = 1, imgPage = 1;
let playInterval = null;

// ── SHARED HELPERS ────────────────────────────────────────────────────────────
function renderPagination(id, current, total, onPage) {
  const el = document.getElementById(id+'-pagination');
  if(!el||total<=1) return;
  let html='';
  html+=`<button class="page-btn" ${current<=1?'disabled':''} onclick="(${onPage.toString()})(${current-1})">‹</button>`;
  const pages=[];
  for(let p=1;p<=total;p++){
    if(p===1||p===total||Math.abs(p-current)<=2) pages.push(p);
    else if(pages[pages.length-1]!=='…') pages.push('…');
  }
  pages.forEach(p=>{
    if(p==='…') html+=`<span style="padding:0 4px;color:var(--text2)">…</span>`;
    else html+=`<button class="page-btn ${p===current?'active':''}" onclick="(${onPage.toString()})(${p})">${p}</button>`;
  });
  html+=`<button class="page-btn" ${current>=total?'disabled':''} onclick="(${onPage.toString()})(${current+1})">›</button>`;
  el.innerHTML=html;
}

// ── DEMO DATA ─────────────────────────────────────────────────────────────────
function demoNumerical(page, limit=50) {
  const cols = ['id','longitude','latitude','year','season','NDSI','NDWI','NDVI','LST_Celsius','elevation','VV','VH','B4','B3','B2','glacier_mask','multiclass_mask'];
  const rows = [];
  for(let i=0;i<limit;i++){
    const r={id: (page-1)*limit + i};
    cols.forEach(c => {
      if(c==='year') r[c]=2018+Math.floor(Math.random()*8);
      else if(c==='season') r[c]=Math.random()>.5?'melt':'winter';
      else if(c==='glacier_mask') r[c]=Math.random()>.65?1:0;
      else if(c==='multiclass_mask') r[c]=Math.floor(Math.random()*4);
      else if(c!=='id') r[c]=+(Math.random()*2-0.5).toFixed(4);
    });
    r.longitude=+(66+Math.random()*19).toFixed(4);
    r.latitude=+(29+Math.random()*11).toFixed(4);
    r.elevation=+(2000+Math.random()*5500).toFixed(0);
    r.LST_Celsius=+(Math.random()*35-20).toFixed(2);
    rows.push(r);
  }
  return {total:28000,page,limit,total_pages:560,columns:cols,data:rows};
}

function demoModelResults() {
  return {
    source:'demo',
    models:{
      'Linear Regression':{task:'regression',type:'ML',MAE:2.41,MSE:9.87,RMSE:3.14,R2:0.71},
      'Random Forest':{task:'regression',type:'ML',MAE:1.23,MSE:3.45,RMSE:1.86,R2:0.89},
      'CNN (FCN)':{task:'segmentation',type:'DL',Accuracy:0.843,Precision:0.831,Recall:0.819,F1:0.825,mIoU:0.721,mDice:0.784,AUC_ROC:0.912,IoU_per_class:[0.821,0.754,0.643,0.668],Dice_per_class:[0.902,0.859,0.783,0.801]},
      'U-Net':{task:'segmentation',type:'DL',Accuracy:0.912,Precision:0.903,Recall:0.894,F1:0.898,mIoU:0.841,mDice:0.893,AUC_ROC:0.967,IoU_per_class:[0.921,0.876,0.781,0.787],Dice_per_class:[0.959,0.934,0.877,0.881]},
      'DeepLabv3+':{task:'segmentation',type:'DL',Accuracy:0.934,Precision:0.921,Recall:0.916,F1:0.918,mIoU:0.874,mDice:0.921,AUC_ROC:0.978,IoU_per_class:[0.943,0.901,0.821,0.831],Dice_per_class:[0.971,0.948,0.902,0.908]}
    }
  };
}

// ── MAP ───────────────────────────────────────────────────────────────────────
const GLACIERS = [
  {name:'Siachen Glacier',lat:35.3162,lng:77.0256,area18:72.5,area25:64.2,elev:4202,ndsi:0.425,ndwi:-0.001,lst:10.0,melt:1.50,class:'glacier'},
  {name:'Baltoro Glacier',lat:35.7423,lng:76.3812,area18:62.1,area25:54.8,elev:4510,ndsi:0.512,ndwi:0.002,lst:8.5,melt:1.15,class:'glacier'},
  {name:'Khumbu Glacier',lat:27.9333,lng:86.8667,area18:48.4,area25:40.1,elev:5200,ndsi:0.385,ndwi:-0.005,lst:12.4,melt:1.95,class:'glacier'},
  {name:'Gangotri Glacier',lat:30.8333,lng:79.0833,area18:35.6,area25:29.8,elev:3850,ndsi:0.410,ndwi:0.001,lst:11.2,melt:1.75,class:'glacier'},
  {name:'Biafo Glacier',lat:35.6833,lng:75.9167,area18:58.2,area25:52.4,elev:4120,ndsi:0.450,ndwi:0.000,lst:9.8,melt:1.35,class:'glacier'},
];

function initMap() {
  const mapEl = document.getElementById('map-container');
  if (!mapEl) return;
  
  mapInstance = L.map('map-container').setView([34.0, 82.0], 5);
  L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
    attribution: '© OpenStreetMap',
    maxZoom: 18
  }).addTo(mapInstance);

  const markers = L.markerClusterGroup({
    showCoverageOnHover: false,
    maxClusterRadius: 50,
    disableClusteringAtZoom: 13
  });

  // Show loading indicator
  const loading = L.control({position: 'topright'});
  loading.onAdd = function() {
    const div = L.DomUtil.create('div', 'map-loading-badge');
    div.innerHTML = '<div class="spinner-small"></div> 🛰️ Syncing 20,000+ Research Points...';
    return div;
  };
  loading.addTo(mapInstance);

  fetch(`${API}/map/points?limit=25000`)
    .then(res => res.json())
    .then(data => {
      data.points.forEach(g => {
        const isGlacier = g.class === 'glacier';
        const m = L.circleMarker([g.lat, g.lng], {
          radius: 6, 
          color: '#fff', 
          fillColor: isGlacier ? '#0d9488' : '#ea580c', 
          fillOpacity: 0.8, 
          weight: 1
        });
        
        m.on('click', () => updateGlacierDetails(g));
        
        m.bindPopup(`
          <div class="map-popup-title">${g.name}</div>
          <div class="map-popup-row"><span>Status:</span><span style="color:${isGlacier?'var(--teal)':'var(--coral)'};font-weight:800">${g.class.toUpperCase()}</span></div>
          <div class="map-popup-row"><span>Elevation:</span><span>${g.elev} m</span></div>
          <div class="map-popup-row"><span>NDSI:</span><span>${g.ndsi}</span></div>
          <div class="map-popup-row"><span>LST:</span><span>${g.lst}°C</span></div>
          <div class="map-popup-row"><span>Melt Rate:</span><span>${g.melt} km²/yr</span></div>
          <div class="map-popup-row"><span>Area (2025):</span><span>${g.area25} km²</span></div>
        `);
        
        markers.addLayer(m);
      });
      
      mapInstance.addLayer(markers);
      if (data.points.length > 0) updateGlacierDetails(data.points[0]);
      mapInstance.removeControl(loading);
    })
    .catch(err => {
      console.error("Map load error:", err);
      // Fallback
      GLACIERS.forEach(g => {
        const m = L.circleMarker([g.lat, g.lng], {
          radius: 8, color: '#fff', fillColor: '#3B8BD4', fillOpacity: 0.9, weight: 2
        });
        m.on('click', () => updateGlacierDetails(g));
        markers.addLayer(m);
      });
      mapInstance.addLayer(markers);
      mapInstance.removeControl(loading);
    });
}

function updateGlacierDetails(g) {
  const set = (id, val) => { const el = document.getElementById(id); if(el) el.textContent = val; };
  set('g-name', g.name);
  set('g-coords', `${g.lat.toFixed(4)}°N, ${g.lng.toFixed(4)}°E`);
  set('g-area-start', `${g.area18} km²`);
  set('g-area-end', `${g.area25} km²`);
  set('g-elev', `${g.elev} m`);
  set('g-ndsi', g.ndsi);
  set('g-ndwi', g.ndwi);
  set('g-lst', `${g.lst}°C`);
  set('g-melt', `${g.melt} km²/yr`);
  
  const lost = g.area18 - g.area25;
  const pct  = (lost / g.area18 * 100).toFixed(1);
  set('g-lost', `-${lost.toFixed(1)} km² (${pct}%)`);
  
  const remain = (100 - pct).toFixed(1);
  set('g-remain-pct', `${remain}%`);
  const fill = document.getElementById('g-progress');
  if(fill) fill.style.width = `${remain}%`;
}

function toggleLayer(id) {
  const toggles = document.querySelectorAll(`.layer-toggle-modern[onclick="toggleLayer('${id}')"]`);
  toggles.forEach(t => t.classList.toggle('active'));
}

function updateOpacity(v) {
  const label = document.getElementById('opacity-label');
  if (label) label.textContent = v + '%';
  // Here you would normally update the opacity of all active analysis layers on the map
}

function updateYear(v) {
  const ids = ['year-label-r', 'top-year', 'g-year'];
  ids.forEach(id => {
    const el = document.getElementById(id);
    if (el) el.textContent = v;
  });
  
  // Update Top Stats based on year (mock logic)
  const stats = {
    '2018': {area:'358.4 km²', temp:'-10.4°C', ndsi:'0.62'},
    '2019': {area:'356.2 km²', temp:'-10.2°C', ndsi:'0.61'},
    '2020': {area:'354.1 km²', temp:'-9.9°C',  ndsi:'0.60'},
    '2021': {area:'352.0 km²', temp:'-9.7°C',  ndsi:'0.59'},
    '2022': {area:'349.8 km²', temp:'-9.5°C',  ndsi:'0.58'},
    '2023': {area:'348.5 km²', temp:'-9.4°C',  ndsi:'0.58'},
    '2024': {area:'347.2 km²', temp:'-9.3°C',  ndsi:'0.57'},
    '2025': {area:'346.1 km²', temp:'-9.2°C',  ndsi:'0.57'},
  };
  const s = stats[v];
  if(s) {
    document.getElementById('top-area').textContent = s.area;
    document.getElementById('top-temp').textContent = s.temp;
    document.getElementById('top-ndsi').textContent = s.ndsi;
  }
}

function playTimelapse() {
  const btn = document.getElementById('play-btn');
  if (playInterval) { clearInterval(playInterval); playInterval=null; btn.textContent='▶ Play'; return; }
  btn.textContent='⏸ Pause';
  let y = parseInt(document.getElementById('year-slider').value);
  playInterval = setInterval(() => {
    y = y >= 2025 ? 2018 : y+1;
    document.getElementById('year-slider').value = y;
    updateYear(y);
    if(y===2025){ clearInterval(playInterval); playInterval=null; btn.textContent='▶ Play'; }
  }, 800);
}

// ── DATA EXPLORER ─────────────────────────────────────────────────────────────
function switchExplorerTab(name, el) {
  document.querySelectorAll('.tab').forEach(t=>t.classList.remove('active'));
  document.querySelectorAll('.tab-panel').forEach(p=>p.classList.remove('active'));
  el.classList.add('active');
  document.getElementById('tab-'+name).classList.add('active');
}

async function loadNumerical() {
  const body = document.getElementById('numerical-body');
  if (!body) return;
  
  const year    = document.getElementById('filter-year').value;
  const season  = document.getElementById('filter-season').value;
  const glacier = document.getElementById('filter-glacier').value;
  body.innerHTML = '<tr><td colspan="20"><div class="loading"><div class="spinner"></div><br>Loading…</div></td></tr>';

  let data;
  try {
    const params = new URLSearchParams({page:numPage,limit:50});
    if(year)    params.append('year',year);
    if(season)  params.append('season',season);
    if(glacier!=='') params.append('glacier',glacier);
    const res = await fetch(`${API}/dataset/numerical?${params}`);
    data = await res.json();
  } catch { data = demoNumerical(numPage); }

  const head = document.getElementById('numerical-head');
  const show  = ['id','status','melt_risk','health_score','longitude','latitude','year','season','NDSI','NDWI','NDVI','LST_Celsius','elevation','VV','VH','B4','B3','B2','glacier_mask','multiclass_mask'];
  head.innerHTML = '<tr><th><input type="checkbox" onclick="toggleAllRows(this)"></th>' + show.map(c=>`<th>${c}</th>`).join('') + '</tr>';

  body.innerHTML = data.data.map(row => {
    const glc = row.glacier_mask;
    const glcColor = glc===1?'rgba(13, 148, 136, 0.05)':'';
    return `<tr style="background:${glcColor}">
      <td style="text-align:center"><input type="checkbox" class="row-check" value="${row.id}"></td>
      ${show.map(c => {
        let v = row[c];
        if(v === undefined || v === null) v = '—';
        if(c==='status') {
          const cls = v==='Critical'?'badge-coral':v==='At Risk'?'badge-amber':'badge-teal';
          v = `<span class="badge ${cls}">${v}</span>`;
        }
        else if(c==='melt_risk') v = `<div style="display:flex;align-items:center;gap:8px;min-width:100px"><div class="progress-bar" style="flex:1;height:6px;margin:0;background:var(--blue-l)"><div class="progress-fill" style="width:${v}%;background:var(--coral)"></div></div><span style="font-size:11px;font-weight:700;color:var(--coral)">${v}%</span></div>`;
        else if(c==='health_score') v = `<div style="display:flex;align-items:center;gap:8px;min-width:100px"><div class="progress-bar" style="flex:1;height:6px;margin:0;background:var(--blue-l)"><div class="progress-fill" style="width:${v}%;background:var(--teal)"></div></div><span style="font-size:11px;font-weight:700;color:var(--teal)">${v}%</span></div>`;
        else if(c==='glacier_mask') v = `<span class="badge ${v===1?'badge-teal':'badge-coral'}" style="min-width:80px;text-align:center">${v===1?'Glacier':'Non-glacier'}</span>`;
        else if(c==='multiclass_mask') {
          const names=['Land','Snow/Ice','Water','Debris'];
          const bgs=['badge-coral','badge-blue','badge-blue','badge-amber'];
          v=`<span class="badge ${bgs[v] || 'badge-gray'}" style="min-width:70px;text-align:center">${names[v] || v}</span>`;
        }
        else if(c==='season') v=`<span class="badge ${v==='melt'?'badge-coral':'badge-blue'}" style="text-transform:capitalize">${v}</span>`;
        else if(typeof v === 'number' && !Number.isInteger(v)) v = v.toFixed(4);
        return `<td>${v}</td>`;
      }).join('')}
    </tr>`;
  }).join('');

  renderPagination('numerical', data.page, data.total_pages, p => { numPage=p; loadNumerical(); });
}

function toggleAllRows(el) {
  document.querySelectorAll('.row-check').forEach(cb => cb.checked = el.checked);
}

function bulkDelete() {
  const selected = Array.from(document.querySelectorAll('.row-check:checked')).map(cb => cb.value);
  if(selected.length === 0) return alert('Select items first');
  if(confirm(`Delete ${selected.length} items?`)) {
    alert(`Successfully deleted ${selected.length} research records.`);
    loadNumerical();
  }
}

function bulkExport() {
  const selected = Array.from(document.querySelectorAll('.row-check:checked')).map(cb => cb.value);
  if(selected.length === 0) return alert('Select items first');
  alert(`Exporting ${selected.length} selected records to research CSV...`);
  window.open(`${API}/export/csv?ids=${selected.join(',')}`,'_blank');
}

function resetFilters() {
  document.getElementById('filter-year').value='';
  document.getElementById('filter-season').value='';
  document.getElementById('filter-glacier').value='';
  numPage=1; loadNumerical();
}

async function loadImages() {
  const grid = document.getElementById('image-grid');
  if (!grid) return;
  
  const count = document.getElementById('img-count');
  grid.innerHTML = '<div class="loading"><div class="spinner"></div><br>Loading images…</div>';

  let data;
  try {
    const res = await fetch(`${API}/dataset/images?page=${imgPage}&limit=24`);
    data = await res.json();
  } catch {
    const items = Array.from({length:24},(_,i)=>({
      filename:`glacier_patch_2022_melt_${((imgPage-1)*24+i).toString().padStart(4, '0')}.tif`,
      url: 'https://via.placeholder.com/256/3B8BD4/FFFFFF?text=Patch',
      year:'2022', season:'melt'
    }));
    data = {total:5250,page:imgPage,total_pages:219,images:items};
  }

  if (count) count.textContent = `Showing ${(imgPage-1)*24+1}–${Math.min(imgPage*24,data.total)} of ${data.total.toLocaleString()} images`;

  grid.innerHTML = data.images.map((img,i) => {
    return `<div class="explorer-item">
      <img src="${img.url}" class="explorer-img" alt="patch">
      <div class="explorer-info">
        <span title="${img.filename}">${img.filename.length > 20 ? img.filename.substring(0,17)+'...' : img.filename}</span>
        <span style="color:var(--teal)">${img.year}</span>
      </div>
    </div>`;
  }).join('');

  renderPagination('image', data.page, data.total_pages, p => { imgPage=p; loadImages(); });
}

// ── RESULTS ───────────────────────────────────────────────────────────────────
async function initResults() {
  if (!document.getElementById('model-cards')) return;
  let data;
  try { const r=await fetch(`${API}/models/results`); data=await r.json(); }
  catch { data = demoModelResults(); }

  renderModelCards(data.models);
  renderComparisonTable(data.models);
  renderPerClassBars(data.models);
  
  // Populate analysis dropdown
  const analysisSelect = document.getElementById('analysis-model-select');
  if (analysisSelect) {
    analysisSelect.innerHTML = Object.keys(data.models).map(name => 
      `<option value="${name}">${name}</option>`
    ).join('');
    updateModelAnalysis();
  }

  loadTrainingCurve();
  loadGlacierStats();
}

function updateModelAnalysis() {
  const modelName = document.getElementById('analysis-model-select').value;
  if (!modelName) return;

  const baseName = modelName.toLowerCase().replace(/ /g, '_');
  const resultsPath = '/results_files';
  
  // Reset all images and "no graph" labels
  for (let i = 1; i <= 6; i++) {
    const img = document.getElementById(`graph-${i}-img`);
    const noGraph = img ? img.nextElementSibling : null;
    if (img) { img.src = ''; img.style.display = 'block'; }
    if (noGraph) noGraph.style.display = 'none';
  }
  ['sample-1-img', 'sample-2-img', 'sample-3-img'].forEach(id => {
    const el = document.getElementById(id);
    if (el) { el.src = ''; el.style.display = 'block'; }
  });

  const isDL = modelName.includes('U-Net') || modelName.includes('CNN') || modelName.includes('DeepLab');
  const isRegression = !modelName.includes('Classifier') && !modelName.includes('Logistic') && !isDL;
  const isBinary = modelName.includes('(Binary)');
  const isMulticlass = modelName.includes('(Multiclass)');

  if (isDL) {
    const dlName = baseName.replace('cnn_(fcn)', 'cnn_fcn').replace('u-net', 'unet').replace('deeplabv3+', 'deeplab');
    document.getElementById('graph-1-img').src = `${resultsPath}/${dlName}_distribution.png`;
    document.getElementById('graph-5-img').src = `${resultsPath}/${dlName}_confusion_matrix.png`;
    document.getElementById('sample-1-img').src = `${resultsPath}/${dlName}_sample_1_pred_vs_actual.png`;
    document.getElementById('sample-2-img').src = `${resultsPath}/${dlName}_sample_2_pred_vs_actual.png`;
    document.getElementById('sample-3-img').src = `${resultsPath}/${dlName}_sample_3_pred_vs_actual.png`;
    // Hide non-relevant
    document.getElementById('graph-2-img').style.display = 'none';
    document.getElementById('graph-3-img').style.display = 'none';
    document.getElementById('graph-4-img').style.display = 'none';
  } 
  else if (isRegression) {
    document.getElementById('graph-1-img').src = `${resultsPath}/${baseName}_dist.png`;
    document.getElementById('graph-2-img').src = `${resultsPath}/${baseName}_pred_vs_actual.png`;
    document.getElementById('graph-3-img').src = `${resultsPath}/${baseName}_residuals.png`;
    
    if (modelName.includes('Ridge') || modelName.includes('Lasso') || modelName.includes('ElasticNet')) {
      document.getElementById('graph-4-img').src = `${resultsPath}/${baseName}_coef.png`;
    } else {
      document.getElementById('graph-4-img').src = `${resultsPath}/${baseName}_feat_imp.png`;
    }
    // Hide non-relevant
    document.getElementById('graph-5-img').style.display = 'none';
    document.getElementById('graph-6-label').parentElement.style.display = 'none';
  }
  else {
    // Classification ML
    const suffix = isBinary ? '_binary' : '_multiclass';
    const cleanBase = baseName.replace('_binary', '').replace('_multiclass', '');
    
    document.getElementById('graph-5-img').src = `${resultsPath}/${cleanBase}${suffix}_cm.png`;
    
    if (modelName.includes('Logistic')) {
      document.getElementById('graph-4-img').src = `${resultsPath}/${cleanBase}${suffix}_coef.png`;
    } else {
      document.getElementById('graph-4-img').src = `${resultsPath}/${cleanBase}${suffix}_feat_imp.png`;
    }
    // Hide non-relevant
    document.getElementById('graph-1-img').style.display = 'none';
    document.getElementById('graph-2-img').style.display = 'none';
    document.getElementById('graph-3-img').style.display = 'none';
    document.getElementById('graph-6-label').parentElement.style.display = 'none';
  }
}

function renderModelCards(models) {
  const TYPE_COLORS = {ML:'badge-coral',DL:'badge-blue'};
  const SHOW_REG   = ['MAE','RMSE','R2'];
  const SHOW_SEG   = ['Accuracy','F1','mIoU','mDice','AUC_ROC'];

  let bestMiou=0, bestName='';
  Object.entries(models).forEach(([name,m])=>{ if((m.mIoU||0)>bestMiou){bestMiou=m.mIoU;bestName=name;} });

  const container = document.getElementById('model-cards');
  container.innerHTML = Object.entries(models).map(([name,m]) => {
    const isBest = name===bestName;
    const show   = m.task==='regression'?SHOW_REG:SHOW_SEG;
    return `<div class="model-card">
      ${isBest?'<div class="best-badge">★ Best</div>':''}
      <div class="model-type"><span class="badge ${TYPE_COLORS[m.type]||'badge-coral'}">${m.type||'ML'}</span></div>
      <div class="model-name">${name}</div>
      ${show.filter(k=>m[k]!==undefined).map(k=>`
        <div class="metric-row"><span>${k}</span><span class="metric-val">${m[k]}</span></div>
      `).join('')}
      <div style="margin-top:10px">
        <div style="display:flex;justify-content:space-between;font-size:11px;color:var(--text2);margin-bottom:4px">
          <span>Performance</span><span>${m.mIoU||m.R2||m.Accuracy||'—'}</span>
        </div>
        <div class="progress-bar"><div class="progress-fill" style="width:${Math.min(100,((m.mIoU||m.R2||m.Accuracy||0)*100).toFixed(0))}%;background:${isBest?'var(--teal)':'var(--blue)'}"></div></div>
      </div>
    </div>`;
  }).join('');
}

function renderComparisonTable(models) {
  const allKeys = new Set();
  Object.values(models).forEach(m => Object.keys(m).forEach(k => {
    if(!['type','task','IoU_per_class','Dice_per_class','confusion_matrix'].includes(k)) allKeys.add(k);
  }));
  const keys = [...allKeys];

  const table = document.getElementById('comparison-table');
  table.innerHTML = `<thead><tr>
    <th>Model</th><th>Type</th><th>Task</th>
    ${keys.map(k=>`<th>${k}</th>`).join('')}
  </tr></thead>
  <tbody>${Object.entries(models).map(([name,m])=>{
    return `<tr class="${name==='DeepLabv3+'?'model-row-deeplab':''}">
      <td><strong>${name}</strong></td>
      <td><span class="badge ${m.type==='DL'?'badge-blue':'badge-coral'}">${m.type}</span></td>
      <td>${m.task}</td>
      ${keys.map(k=>{
        const v = m[k];
        if(v===undefined) return '<td>—</td>';
        const isTop = typeof v==='number' && v>0.9;
        return `<td class="${isTop?'best':''}">${v}</td>`;
      }).join('')}
    </tr>`;
  }).join('')}</tbody>`;
}

function renderPerClassBars(models) {
  const classes  = ['Land','Snow/Ice','Water','Debris ice'];
  const colors   = ['#888780','#B5D4F4','#3B8BD4','#FAC775'];
  const deeplab  = models['DeepLabv3+'] || Object.values(models).pop();

  if (document.getElementById('iou-bars')) {
    ['iou','dice'].forEach(metric => {
      const vals = metric==='iou' ? (deeplab.IoU_per_class||[.943,.901,.821,.831]) : (deeplab.Dice_per_class||[.971,.948,.902,.908]);
      document.getElementById(`${metric}-bars`).innerHTML = classes.map((c,i)=>`
        <div class="class-bar-row">
          <div class="class-bar-label">${c}</div>
          <div class="class-bar-track"><div class="class-bar-fill" style="width:${(vals[i]*100).toFixed(0)}%;background:${colors[i]}"></div></div>
          <div class="class-bar-val">${vals[i]}</div>
        </div>
      `).join('');
    });
  }
}

async function loadTrainingCurve() {
  const modelEl = document.getElementById('curve-model');
  if (!modelEl) return;
  const model = modelEl.value;
  let data;
  try { const r=await fetch(`${API}/models/training_history/${model}`); data=await r.json(); }
  catch {
    const n=80;
    const e=Array.from({length:n},(_,i)=>i+1);
    const tl=e.map(i=>+(1.1*Math.exp(-0.04*i)+Math.random()*.03).toFixed(4));
    const vl=tl.map(v=>+(v*1.05).toFixed(4));
    const tm=e.map(i=>+Math.min(.94,.3+.008*i+Math.random()*.01-.005).toFixed(4));
    const vm=tm.map(v=>+(v*.98).toFixed(4));
    data={history:e.map((ep,i)=>({epoch:ep,train_loss:tl[i],val_loss:vl[i],train_mIoU:tm[i],val_mIoU:vm[i]}))};
  }
  if(curveChart) curveChart.destroy();
  const epochs = data.history.map(h=>h.epoch);
  curveChart = new Chart(document.getElementById('curve-chart'),{
    type:'line',
    data:{
      labels:epochs,
      datasets:[
        {label:'Train mIoU',data:data.history.map(h=>h.train_mIoU),borderColor:'#3B8BD4',tension:.3,pointRadius:0},
        {label:'Val mIoU',  data:data.history.map(h=>h.val_mIoU),  borderColor:'#1D9E75',tension:.3,pointRadius:0},
        {label:'Train Loss',data:data.history.map(h=>h.train_loss),borderColor:'#E8593C',tension:.3,pointRadius:0,yAxisID:'y2'},
        {label:'Val Loss',  data:data.history.map(h=>h.val_loss),  borderColor:'#FAC775',tension:.3,pointRadius:0,yAxisID:'y2'},
      ]
    },
    options:{responsive:true,maintainAspectRatio:false,plugins:{legend:{labels:{font:{size:11}}}},
      scales:{y:{title:{display:true,text:'mIoU'}},y2:{position:'right',title:{display:true,text:'Loss'},grid:{drawOnChartArea:false}}}}
  });
}

async function loadGlacierStats() {
  const years  = [2018,2019,2020,2021,2022,2023,2024,2025];
  // Simulated stats for the full research ROI
  const area   = [124500, 122800, 121150, 119400, 117600, 115710, 113980, 112150];
  const temp   = [-4.1, -3.98, -3.82, -3.71, -3.63, -3.51, -3.38, -3.24];
  const water  = [4200, 4310, 4440, 4590, 4720, 4870, 5010, 5180];

  const insights = [
    {val:'−12,350 km²',label:'Total area lost (2018–25)',cls:'coral'},
    {val:'−9.92%',    label:'Percentage area loss',cls:'coral'},
    {val:'+980 km²',  label:'New proglacial lakes',cls:'blue'},
    {val:'+0.86°C',   label:'Mean LST increase',cls:'amber'},
  ];
  const container = document.getElementById('insight-cards');
  if (container) {
    container.innerHTML = insights.map(ins=>`
      <div class="insight-card ${ins.cls}">
        <div class="insight-val">${ins.val}</div>
        <div class="insight-label">${ins.label}</div>
      </div>
    `).join('');
  }

  if(document.getElementById('area-chart')) {
    if(areaChart) areaChart.destroy();
    areaChart = new Chart(document.getElementById('area-chart'),{
      type:'line',
      data:{labels:years.map(String),datasets:[{label:'Glacier area (km²)',data:area,borderColor:'#3B8BD4',backgroundColor:'rgba(59,139,212,.1)',fill:true,tension:.4,pointRadius:5}]},
      options:{responsive:true,maintainAspectRatio:false,scales:{y:{title:{display:true,text:'km²'}}},plugins:{legend:{display:false}}}
    });
  }

  if(document.getElementById('lst-chart')) {
    if(lstChart) lstChart.destroy();
    lstChart = new Chart(document.getElementById('lst-chart'),{
      type:'line',
      data:{labels:years.map(String),datasets:[{label:'Mean LST (°C)',data:temp,borderColor:'#E8593C',backgroundColor:'rgba(216,90,48,.1)',fill:true,tension:.4,pointRadius:5}]},
      options:{responsive:true,maintainAspectRatio:false,scales:{y:{title:{display:true,text:'°C'}}},plugins:{legend:{display:false}}}
    });
  }

  if(document.getElementById('water-chart')) {
    if(waterChart) waterChart.destroy();
    waterChart = new Chart(document.getElementById('water-chart'),{
      type:'bar',
      data:{labels:years.map(String),datasets:[{label:'Water bodies (km²)',data:water,backgroundColor:'rgba(59,139,212,.7)',borderRadius:4}]},
      options:{responsive:true,maintainAspectRatio:false,scales:{y:{title:{display:true,text:'km²'}}},plugins:{legend:{display:false}}}
    });
  }
}

async function exportCSV() {
  window.open(`${API}/export/csv`,'_blank');
}

async function exportPDF() {
  window.open(`${API}/export/pdf`,'_blank');
}

// ── ANALYTICS ─────────────────────────────────────────────────────────────────
function initAnalytics() {
  if (!document.getElementById('spectral-trend-chart')) return;
  const years = [2018,2019,2020,2021,2022,2023,2024,2025];
  
  if(spectralChart) spectralChart.destroy();
  spectralChart = new Chart(document.getElementById('spectral-trend-chart'), {
    type: 'line',
    data: {
      labels: years,
      datasets: [
        {label: 'NDSI (Snow)', data: [0.72, 0.70, 0.68, 0.65, 0.63, 0.61, 0.58, 0.55], borderColor: '#3B8BD4', tension: 0.4},
        {label: 'NDVI (Veg)', data: [0.12, 0.14, 0.15, 0.18, 0.21, 0.23, 0.25, 0.28], borderColor: '#1D9E75', tension: 0.4}
      ]
    },
    options: {responsive: true, maintainAspectRatio: false}
  });

  if(elevChart) elevChart.destroy();
  elevChart = new Chart(document.getElementById('elev-dist-chart'), {
    type: 'bar',
    data: {
      labels: ['2k-3k', '3k-4k', '4k-5k', '5k-6k', '6k+'],
      datasets: [{label: 'Glacier Area %', data: [5, 15, 45, 25, 10], backgroundColor: '#534AB7'}]
    },
    options: {responsive: true, maintainAspectRatio: false}
  });

  if(scatterChart) scatterChart.destroy();
  const scatterData = Array.from({length: 50}, () => ({x: Math.random()*15 - 5, y: Math.random()*5000 + 1000}));
  scatterChart = new Chart(document.getElementById('lst-area-scatter'), {
    type: 'scatter',
    data: { datasets: [{label: 'Glaciers', data: scatterData, backgroundColor: 'rgba(216,90,48,0.6)'}] },
    options: {responsive: true, maintainAspectRatio: false, scales: {x: {title: {display:true, text: 'LST (°C)'}}, y: {title: {display:true, text: 'Area (km²)'}}}}
  });

  if(precipChart) precipChart.destroy();
  precipChart = new Chart(document.getElementById('precip-chart'), {
    type: 'line',
    data: { labels: years, datasets: [{label: 'Annual Snowfall (mm)', data: [850, 920, 780, 810, 740, 690, 720, 650], borderColor: '#EF9F27', fill: true, backgroundColor: 'rgba(239,159,39,0.1)'}] },
    options: {responsive: true, maintainAspectRatio: false}
  });
}

function updateComparison() {
  const g1Select = document.getElementById('compare-g1');
  const g2Select = document.getElementById('compare-g2');
  const g1Name = g1Select.selectedOptions[0].text.split(' ')[0];
  const g2Name = g2Select.selectedOptions[0].text.split(' ')[0];
  
  document.getElementById('g1-name').textContent = g1Name;
  document.getElementById('g2-name').textContent = g2Name;

  // Simulate unique metrics per glacier
  const metrics = {
    'baltoro': {melt: '-1.4%', health: '82.5'},
    'khumbu':  {melt: '-2.1%', health: '64.2'},
    'siachen': {melt: '-0.9%', health: '88.1'},
    'gangotri':{melt: '-2.8%', health: '52.4'}
  };

  const m1 = metrics[g1Select.value] || metrics['baltoro'];
  const m2 = metrics[g2Select.value] || metrics['khumbu'];

  document.getElementById('g1-melt').textContent = m1.melt + ' / yr';
  document.getElementById('g1-health').textContent = m1.health;
  document.getElementById('g2-melt').textContent = m2.melt + ' / yr';
  document.getElementById('g2-health').textContent = m2.health;
}

// ── SANDBOX ───────────────────────────────────────────────────────────────────
function initSandbox() {
  const epochsInput = document.getElementById('sb-epochs');
  if (!epochsInput) return;
  epochsInput.oninput = (e) => document.getElementById('sb-epoch-val').textContent = e.target.value;
  document.getElementById('sb-lr').oninput = (e) => {
    const lrs = [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1];
    const idx = Math.floor((e.target.value / 100) * (lrs.length - 1));
    document.getElementById('sb-lr-val').textContent = lrs[idx];
  };
}

function runSandboxTraining() {
  const epochs = parseInt(document.getElementById('sb-epochs').value);
  const model = document.getElementById('sb-model').value;
  const impact = document.getElementById('sb-impact');
  
  impact.innerHTML = `<div class="loading"><div class="spinner"></div><br>Simulating ${model.toUpperCase()} training for ${epochs} steps...</div>`;
  
  let currentStep = 0;
  if(sbLossChart) sbLossChart.destroy();
  if(sbAccChart) sbAccChart.destroy();
  
  sbLossChart = new Chart(document.getElementById('sb-loss-chart'), {
    type: 'line',
    data: { labels: [], datasets: [{label: 'Loss', data: [], borderColor: '#E8593C', pointRadius: 0}] },
    options: {responsive: true, maintainAspectRatio: false, animation: false}
  });
  
  sbAccChart = new Chart(document.getElementById('sb-acc-chart'), {
    type: 'line',
    data: { labels: [], datasets: [{label: 'Accuracy', data: [], borderColor: '#1D9E75', pointRadius: 0}] },
    options: {responsive: true, maintainAspectRatio: false, animation: false}
  });

  const interval = setInterval(() => {
    currentStep++;
    const loss = 1.0 * Math.exp(-0.05 * currentStep) + Math.random() * 0.05;
    const acc = Math.min(0.98, 0.4 + 0.01 * currentStep + Math.random() * 0.02);
    
    sbLossChart.data.labels.push(currentStep);
    sbLossChart.data.datasets[0].data.push(loss);
    sbAccChart.data.labels.push(currentStep);
    sbAccChart.data.datasets[0].data.push(acc);
    
    sbLossChart.update();
    sbAccChart.update();
    
    if(currentStep >= epochs) {
      clearInterval(interval);
      impact.innerHTML = `<strong>Simulation Complete!</strong><br>Final Loss: ${loss.toFixed(4)} | Final Accuracy: ${(acc*100).toFixed(2)}%<br>Hyperparameters used: Epochs=${epochs}, Model=${model.toUpperCase()}`;
    }
  }, 50);
}

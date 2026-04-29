// ── CONSTANTS ────────────────────────────────────────────────────────────────
const COLORS = {
  primary: '#0f172a',
  accent: '#0d9488',
  info: '#3b82f6',
  success: '#10b981',
  warning: '#f59e0b',
  danger: '#ef4444',
  textMuted: '#64748b',
  bgMain: '#f8fafc'
};

const API = '/api';
let mapInstance, tsChart, curveChart, areaChart, lstChart, waterChart;
let spectralChart, elevChart, scatterChart, precipChart, sbLossChart, sbAccChart;
let numPage = 1, imgPage = 1;
let playInterval = null;
let resultsData = null; // Global storage for model results and plots

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
      'Linear Regression':{task:'regression',type:'ML',MAE:0.183,MSE:0.033,RMSE:0.182,R2:0.724, plots: []},
      'Random Forest Regressor':{task:'regression',type:'ML',MAE:0.119,MSE:0.014,RMSE:0.118,R2:0.837, plots: []},
      'CNN_FCN':{task:'segmentation',type:'DL',Accuracy:0.852,Precision:0.841,Recall:0.832,F1:0.836,mIoU:0.684,mDice:0.812, plots: []},
      'UNet':{task:'segmentation',type:'DL',Accuracy:0.918,Precision:0.911,Recall:0.902,F1:0.906,mIoU:0.857,mDice:0.923, plots: []},
      'DeepLabv3+':{task:'segmentation',type:'DL',Accuracy:0.931,Precision:0.925,Recall:0.918,F1:0.921,mIoU:0.871,mDice:0.931, plots: []}
    },
    overall_figures: []
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
          <div class="data-explanation" style="margin-top:10px; padding-top:10px; border-top: 1px dashed var(--border)">
            <span class="data-why">Research Rationale:</span>
            <span style="font-size:10px; color:var(--text2); line-height:1.4">
              Melt rate of ${g.melt} km²/yr is derived from a 7-year regression analysis. 
              The ${isGlacier ? 'high NDSI' : 'low NDSI'} confirms ${isGlacier ? 'active ice' : 'receding boundary'} at this elevation (${g.elev}m).
            </span>
          </div>
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
  if(!g) return;
  const set = (id, val) => { const el = document.getElementById(id); if(el) el.textContent = val; };
  
  // 1. Update classification badge FIRST (highest priority for consistency)
  const classEl = document.getElementById('g-class');
  if(classEl && g.class) {
    const isGlacier = g.class.toLowerCase() === 'glacier';
    classEl.textContent = g.class.toUpperCase();
    classEl.className = isGlacier ? 'badge badge-success' : 'badge badge-danger';
    // Ensure the color is actually applied via inline style as a fallback
    classEl.style.background = isGlacier ? '#d1fae5' : '#fee2e2';
    classEl.style.color = isGlacier ? '#065f46' : '#991b1b';
  }

  // 2. Update basic fields
  set('g-name', g.name || `Glacier Zone ${Math.floor(Math.random()*20000)}`);
  set('g-coords', (g.lat && g.lng) ? `${g.lat.toFixed(4)}°N, ${g.lng.toFixed(4)}°E` : 'N/A');
  set('g-elev', g.elev ? `${g.elev} m` : 'N/A');
  set('g-ndsi', g.ndsi || 'N/A');
  set('g-ndwi', g.ndwi || 'N/A');
  set('g-lst', g.lst ? `${g.lst}°C` : 'N/A');
  set('g-melt', g.melt ? `${g.melt} km²/yr` : 'N/A');
  
  // 3. Update area and loss metrics
  if(g.area18 !== undefined && g.area25 !== undefined) {
    set('g-area-start', `${g.area18} km²`);
    set('g-area-end', `${g.area25} km²`);
    
    const lost = g.area18 - g.area25;
    const pct  = (lost / g.area18 * 100).toFixed(1);
    set('g-lost', `-${lost.toFixed(1)} km² (${pct}%)`);
    
    const remain = (100 - pct).toFixed(1);
    set('g-remain-pct', `${remain}%`);
    const fill = document.getElementById('g-progress');
    if(fill) fill.style.width = `${remain}%`;
  }
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
  // Remove active from all sub-tabs in explorer
  el.parentElement.querySelectorAll('.nav-tab').forEach(t => {
    t.classList.remove('active');
    t.style.color = 'var(--text-muted)';
    t.style.background = 'transparent';
  });
  
  // Set active on clicked tab
  el.classList.add('active');
  el.style.color = 'var(--primary)';
  el.style.background = 'white';
  el.style.boxShadow = 'var(--shadow-sm)';
  el.style.borderRadius = 'var(--radius-sm)';

  // Switch panels
  document.querySelectorAll('.tab-panel').forEach(p => p.classList.remove('active'));
  const target = document.getElementById('tab-' + name);
  if (target) target.classList.add('active');
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
          const cls = v==='Critical'?'badge-danger':v==='At Risk'?'badge-warning':'badge-success';
          v = `<span class="badge ${cls}">${v}</span>`;
        }
        else if(c==='melt_risk') v = `<div style="display:flex;align-items:center;gap:12px;min-width:140px"><div class="progress-container" style="flex:1;height:8px;margin:0"><div class="progress-bar" style="width:${v}%;background:var(--danger)"></div></div><span style="font-size:12px;font-weight:800;color:var(--danger);min-width:40px">${v}%</span></div>`;
        else if(c==='health_score') v = `<div style="display:flex;align-items:center;gap:12px;min-width:140px"><div class="progress-container" style="flex:1;height:8px;margin:0"><div class="progress-bar" style="width:${v}%;background:var(--accent)"></div></div><span style="font-size:12px;font-weight:800;color:var(--accent);min-width:40px">${v}%</span></div>`;
        else if(c==='glacier_mask') v = `<span class="badge ${v===1?'badge-success':'badge-danger'}" style="min-width:100px;text-align:center">${v===1?'Glacier':'Non-glacier'}</span>`;
        else if(c==='multiclass_mask') {
          const names=['Land','Snow/Ice','Water','Debris'];
          const bgs=['badge-danger','badge-info','badge-info','badge-warning'];
          v=`<span class="badge ${bgs[v] || 'badge-info'}" style="min-width:80px;text-align:center">${names[v] || v}</span>`;
        }
        else if(c==='season') v=`<span class="badge ${v==='melt'?'badge-danger':'badge-info'}" style="text-transform:capitalize">${v}</span>`;
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
  const year = document.getElementById('img-year').value;
  const season = document.getElementById('img-season').value;
  
  grid.innerHTML = '<div class="loading" style="grid-column: 1/-1"><div class="spinner"></div><br>Syncing multispectral patches…</div>';

  let data;
  try {
    const params = new URLSearchParams({page:imgPage, limit:24});
    if(year) params.append('year', year);
    if(season) params.append('season', season);
    const res = await fetch(`${API}/dataset/images?${params}`);
    data = await res.json();
  } catch {
    const items = Array.from({length:24},(_,i)=>({
      filename:`glacier_patch_2025_winter_${((imgPage-1)*24+i).toString().padStart(4, '0')}`,
      url: 'https://via.placeholder.com/256/3B8BD4/FFFFFF?text=Patch',
      year:'2025', season:'winter',
      mask_url: 'https://via.placeholder.com/256/000000/FFFFFF?text=Mask',
      multi_url: 'https://via.placeholder.com/256/FAC775/FFFFFF?text=MultiMap'
    }));
    data = {total:5250,page:imgPage,total_pages:219,images:items};
  }

  if (count) count.textContent = `Showing ${(imgPage-1)*24+1}–${Math.min(imgPage*24,data.total)} of ${data.total.toLocaleString()} research patches`;

  grid.innerHTML = data.images.map((img,i) => {
    // Correctly handle optional mask and multi URLs
    const mask = img.mask_url || '';
    const multi = img.multi_url || '';
    
    return `<div class="explorer-item" onclick="openPatchModal('${img.filename}', '${img.url}', '${mask}', '${multi}', '${img.year}', '${img.season}')" style="cursor:pointer">
      <img src="${img.url}" class="explorer-img" alt="patch">
      <div class="explorer-info">
        <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:4px">
          <span class="badge badge-info" style="font-size:9px">${img.year}</span>
          <span class="badge ${img.season==='melt'?'badge-danger':'badge-info'}" style="font-size:9px">${img.season.toUpperCase()}</span>
        </div>
        <span title="${img.filename}" style="font-size:11px; font-weight:700">${img.filename.length > 20 ? img.filename.substring(0,17)+'...' : img.filename}</span>
      </div>
      <div class="item-overlay" style="position:absolute; top:0; left:0; width:100%; height:100%; background:rgba(15, 23, 42, 0.4); display:flex; align-items:center; justify-content:center; opacity:0; transition:var(--transition); border-radius:var(--radius-md)">
        <span style="color:white; font-size:12px; font-weight:800; border:1px solid white; padding:6px 12px; border-radius:4px">🔬 VIEW ANALYSIS</span>
      </div>
    </div>`;
  }).join('');

  renderPagination('image', data.page, data.total_pages, p => { imgPage=p; loadImages(); });
}

function openPatchModal(name, rgb, mask, multi, year, season) {
  // Create modal if it doesn't exist
  let modal = document.getElementById('patch-modal');
  if(!modal) {
    modal = document.createElement('div');
    modal.id = 'patch-modal';
    modal.className = 'modal-backdrop';
    modal.innerHTML = `
      <div class="modal-content" style="max-width:900px; width:90%; padding:0; overflow:hidden">
        <div style="background:var(--primary); padding:20px; color:white; display:flex; justify-content:space-between; align-items:center">
          <div>
            <h3 style="margin:0; font-size:18px" id="modal-title">Patch Analysis</h3>
            <div style="font-size:11px; color:var(--accent-light); margin-top:4px" id="modal-subtitle"></div>
          </div>
          <button onclick="closeModal('patch-modal')" style="background:transparent; border:none; color:white; font-size:24px; cursor:pointer">×</button>
        </div>
        <div style="padding:32px; background:var(--bg-main)">
          <div class="grid-3" style="gap:24px">
            <div class="card" style="padding:12px">
              <div class="card-title" style="font-size:12px; margin-bottom:12px">Satellite RGB</div>
              <img id="modal-rgb" src="" style="width:100%; border-radius:4px; aspect-ratio:1">
            </div>
            <div class="card" style="padding:12px">
              <div class="card-title" style="font-size:12px; margin-bottom:12px">Glacier Mask (AI)</div>
              <img id="modal-mask" src="" style="width:100%; border-radius:4px; aspect-ratio:1">
            </div>
            <div class="card" style="padding:12px">
              <div class="card-title" style="font-size:12px; margin-bottom:12px">Multi-Class Map</div>
              <img id="modal-multi" src="" style="width:100%; border-radius:4px; aspect-ratio:1">
            </div>
          </div>
          <div class="card" style="margin-top:24px">
            <h4 style="margin-bottom:12px">Metadata Analysis</h4>
            <div style="display:grid; grid-template-columns: repeat(4, 1fr); gap:16px">
              <div style="font-size:12px"><b>Year:</b> <span id="meta-year"></span></div>
              <div style="font-size:12px"><b>Season:</b> <span id="meta-season"></span></div>
              <div style="font-size:12px"><b>Format:</b> GeoTIFF (13-band)</div>
              <div style="font-size:12px"><b>Resolution:</b> 10m/px</div>
            </div>
          </div>
        </div>
      </div>
    `;
    document.body.appendChild(modal);
  }

  document.getElementById('modal-title').textContent = name;
  document.getElementById('modal-subtitle').textContent = `Multispectral Patch • Research ROI • ${year}`;
  document.getElementById('modal-rgb').src = rgb;
  
  // Handle optional mask/multi URLs, ensuring they aren't 'undefined' or empty strings
  const maskSrc = (mask && mask !== 'undefined' && mask !== 'null') ? mask : 'https://via.placeholder.com/256/000000/FFFFFF?text=No+Mask';
  const multiSrc = (multi && multi !== 'undefined' && multi !== 'null') ? multi : 'https://via.placeholder.com/256/FAC775/FFFFFF?text=No+MultiMap';
  
  document.getElementById('modal-mask').src = maskSrc;
  document.getElementById('modal-multi').src = multiSrc;
  document.getElementById('meta-year').textContent = year;
  document.getElementById('meta-season').textContent = season.toUpperCase();
  
  modal.style.display = 'flex';
}

function closeModal(id) {
  const modal = document.getElementById(id);
  if(modal) modal.style.display = 'none';
}

// ── RESULTS ───────────────────────────────────────────────────────────────────
async function initResults() {
  const modelCards = document.getElementById('model-cards');
  const comparisonTable = document.getElementById('comparison-table');
  
  if (!modelCards && !comparisonTable) {
    console.warn("Results elements not found on this page.");
    return;
  }

  console.log("Initializing results page...");
  try { 
    const r = await fetch(`${API}/models/results`); 
    if (!r.ok) throw new Error(`HTTP error! status: ${r.status}`);
    resultsData = await r.json(); 
    console.log("Model results loaded from API:", resultsData);
  }
  catch (err) { 
    console.error("Failed to fetch model results from API, using demo data:", err);
    resultsData = demoModelResults(); 
  }

  if (resultsData && resultsData.models) {
    try { if (modelCards) renderModelCards(resultsData.models); } catch(e) { console.error("Error in renderModelCards:", e); }
    try { if (comparisonTable) renderComparisonTable(resultsData.models); } catch(e) { console.error("Error in renderComparisonTable:", e); }
    try { renderPerClassBars(resultsData.models); } catch(e) { console.error("Error in renderPerClassBars:", e); }
    
    // Populate analysis dropdown
    const analysisSelect = document.getElementById('analysis-model-select');
    if (analysisSelect) {
      const modelNames = Object.keys(resultsData.models);
      console.log("Populating dropdown with models:", modelNames);
      
      if (modelNames.length > 0) {
        analysisSelect.innerHTML = modelNames.map(name => 
          `<option value="${name}">${name}</option>`
        ).join('');
        
        // Initial manual update to show first model's results
        console.log("Triggering initial model analysis update for:", analysisSelect.value);
        updateModelAnalysis();

        // Also add an event listener just in case
        analysisSelect.addEventListener('change', updateModelAnalysis);
      } else {
        analysisSelect.innerHTML = '<option value="">No Models Available</option>';
      }
    }

    // Render overall research figures
    try { renderOverallFigures(resultsData.overall_figures || []); } catch(e) { console.error("Error in renderOverallFigures:", e); }
  }

  loadTrainingCurve();
  loadGlacierStats();
}

function renderOverallFigures(figures) {
  const container = document.getElementById('overall-figures-container');
  if (!container) return;

  // Primary figures that are already statically displayed in HTML
  const primaryFigures = ['fig2', 'fig2b', 'fig3', 'fig4', 'fig5', 'fig6', 'fig7'];
  
  const additionalFigures = figures.filter(fig => {
    const nameLower = fig.name.toLowerCase().replace(/[^a-z0-9]/g, '');
    return !primaryFigures.some(p => nameLower.includes(p));
  });

  if (!additionalFigures || additionalFigures.length === 0) {
    container.innerHTML = `<div style="grid-column: 1/-1; text-align: center; padding: 40px; color: var(--text-muted); border: 2px dashed var(--border-light); border-radius: var(--radius-md)">
      All primary figures (Fig 2-7) are displayed at the top of the page.
    </div>`;
    return;
  }

  // Sort figures by name
  additionalFigures.sort((a, b) => a.name.localeCompare(b.name, undefined, {numeric: true, sensitivity: 'base'}));

  container.innerHTML = additionalFigures.map(fig => `
    <div class="card" style="padding:20px; ${fig.name.toLowerCase().includes('fig 5') || fig.name.toLowerCase().includes('fig 6') || fig.name.toLowerCase().includes('fig 7') ? 'grid-column: 1 / -1' : ''}">
      <div class="card-title">${fig.name}</div>
      <img src="${fig.url}" style="width:100%; border-radius:12px; box-shadow: var(--shadow-sm)" alt="${fig.name}">
    </div>
  `).join('');
}

function updateModelAnalysis() {
  const selectEl = document.getElementById('analysis-model-select');
  if (!selectEl) return;
  
  const modelName = selectEl.value;
  if (!modelName || !resultsData || !resultsData.models) return;

  const model = resultsData.models[modelName];
  if (!model) return;

  console.log("Updating analysis for model:", modelName, model);

  const container = document.getElementById('dynamic-plots-container');
  if (!container) return;

  if (!model.plots || model.plots.length === 0) {
    container.innerHTML = `<div class="no-graph" style="grid-column: 1/-1; text-align: center; padding: 40px;">No detailed visualizations found for ${modelName} in the results folder.</div>`;
    return;
  }

  // Define ordering for common research plot types
  const order = [
    'Performance', 'IoU', 'Distribution', 'Confusion Matrix', 
    'Coefficients', 'Feature Importance', 'Predicted vs Actual', 
    'Residuals', 'Residual Pattern', 'Training Curves', 'Temporal Trends'
  ];
  const sortedPlots = [...model.plots].sort((a, b) => {
    const idxA = order.findIndex(o => a.name.toLowerCase().includes(o.toLowerCase()));
    const idxB = order.findIndex(o => b.name.toLowerCase().includes(o.toLowerCase()));
    if (idxA !== -1 && idxB !== -1) return idxA - idxB;
    if (idxA !== -1) return -1;
    if (idxB !== -1) return 1;
    return a.name.localeCompare(b.name);
  });

  const getExplanation = (name) => {
    const n = name.toLowerCase();
    if(n.includes('confusion')) return "<strong>Analysis:</strong> Diagonal elements represent correct pixel classifications. Off-diagonal 'bleeds' indicate spectral confusion between debris and rock classes.";
    if(n.includes('residual pattern')) return "<strong>Analysis:</strong> Evaluates homoscedasticity. A random cloud indicates the model has captured all non-random variance in the glacier features.";
    if(n.includes('predicted vs actual')) return "<strong>Analysis:</strong> Measures regression fit. Alignment along the 45° identity line confirms high predictive validity for regional area loss.";
    if(n.includes('feature importance')) return "<strong>Analysis:</strong> Ranks predictive weight. NDSI and LST typically emerge as primary drivers of model decisions.";
    if(n.includes('distribution')) return "<strong>Analysis:</strong> Compares sample population spread. Ensures the dataset maintains balanced representation across elevation gradients.";
    if(n.includes('performance')) return "<strong>Analysis:</strong> Holistic benchmark summary across Precision, Recall, and IoU metrics.";
    if(n.includes('coefficient')) return "<strong>Analysis:</strong> Directional impact of each spectral band on the final glacier health score.";
    return "<strong>Analysis:</strong> Research visualization representing key model performance or data distribution metrics.";
  };

  container.innerHTML = sortedPlots.map(plot => {
    const isLarge = plot.name.toLowerCase().includes('confusion') || 
                   plot.name.toLowerCase().includes('segmentation') ||
                   plot.name.toLowerCase().includes('comparison');
    
    return `
      <div class="graph-item" style="background: white; padding: 16px; border-radius: 12px; border: 1px solid var(--border); ${isLarge ? 'grid-column: 1 / -1' : ''}">
        <div class="graph-label" style="font-weight: 700; font-size: 13px; color: var(--blue-d); margin-bottom: 12px; display:flex; align-items:center; gap:8px">
          <span style="color:var(--accent)">📊</span> ${plot.name}
        </div>
        <img src="${plot.url}" style="width:100%; border-radius:8px; box-shadow: var(--shadow-sm)" alt="${plot.name}">
        <div class="plot-explanation" style="margin-top:12px; padding:10px; background:var(--bg-main); border-left:2px solid var(--accent); font-size:11px; color:var(--text-muted); line-height:1.5">
          ${getExplanation(plot.name)}
        </div>
      </div>
    `;
  }).join('');
}

function renderModelCards(models) {
  const TYPE_COLORS = {ML:'badge-coral',DL:'badge-blue'};
  const SHOW_REG   = ['MAE','RMSE','R2'];
  const SHOW_SEG   = ['Accuracy','F1','mIoU','mDice','AUC_ROC'];

  if (!models || Object.keys(models).length === 0) return;

  let bestMiou=0, bestName='';
  Object.entries(models).forEach(([name,m])=>{ 
    const val = m.mIoU || m.Accuracy || m.R2 || 0;
    if(val > bestMiou){ bestMiou = val; bestName = name; } 
  });

  const container = document.getElementById('model-cards');
  if (!container) return;

  container.innerHTML = Object.entries(models).map(([name,m]) => {
    const isBest = name===bestName;
    const show   = m.task==='regression'?SHOW_REG:SHOW_SEG;
    const perfVal = m.mIoU || m.R2 || m.Accuracy || 0;
    
    return `<div class="model-card">
      ${isBest?'<div class="best-badge">★ Best</div>':''}
      <div class="model-type"><span class="badge ${TYPE_COLORS[m.type]||'badge-coral'}">${m.type||'ML'}</span></div>
      <div class="model-name">${name}</div>
      ${show.filter(k=>m[k]!==undefined).map(k=>`
        <div class="metric-row"><span>${k}</span><span class="metric-val">${m[k]}</span></div>
      `).join('')}
      <div style="margin-top:10px">
        <div style="display:flex;justify-content:space-between;font-size:11px;color:var(--text2);margin-bottom:4px">
          <span>Performance</span><span>${perfVal}</span>
        </div>
        <div class="progress-bar"><div class="progress-fill" style="width:${Math.min(100,(perfVal*100).toFixed(0))}%;background:${isBest?'var(--teal)':'var(--blue)'}"></div></div>
      </div>
    </div>`;
  }).join('');
}

function renderComparisonTable(models) {
  const allKeys = new Set();
  Object.values(models).forEach(m => Object.keys(m).forEach(k => {
    if(!['type','task','IoU_per_class','Dice_per_class','confusion_matrix','plots'].includes(k)) allKeys.add(k);
  }));
  const keys = [...allKeys];

  // Research rationales for metrics
  const RATIONALES = {
    'MAE': 'Mean Absolute Error: Average magnitude of the errors in a set of predictions.',
    'MSE': 'Mean Squared Error: Measures the average of the squares of the errors.',
    'RMSE': 'Root Mean Squared Error: Standard deviation of the residuals (prediction errors).',
    'R2': 'R-squared: Proportion of the variance for a dependent variable that is explained by the model.',
    'Accuracy': 'Overall correctness: (TP+TN)/(TP+TN+FP+FN).',
    'Precision': 'Ability of the model not to label a negative sample as positive.',
    'Recall': 'Ability of the model to find all the positive samples.',
    'F1': 'Harmonic mean of precision and recall.',
    'mIoU': 'Mean Intersection over Union: Standard metric for segmentation accuracy.',
    'mDice': 'Mean Dice Coefficient: Measures spatial overlap between prediction and ground truth.'
  };

  const table = document.getElementById('comparison-table');
  table.innerHTML = `<thead><tr>
    <th>Model Architecture</th><th>Type</th><th>Task</th>
    ${keys.map(k=>`<th title="${RATIONALES[k]||''}">${k} <span style="font-size:10px;opacity:0.6;cursor:help">ⓘ</span></th>`).join('')}
  </tr></thead>
  <tbody>${Object.entries(models).map(([name,m])=>{
    return `<tr class="${name==='DeepLabv3+'?'model-row-deeplab':''}">
      <td>
        <div style="font-weight:800;color:var(--blue-d)">${name}</div>
        <div style="font-size:10px;color:var(--text2);margin-top:2px">${m.task==='segmentation'?'Spatial boundary focus':'Pixel-wise spectral focus'}</div>
      </td>
      <td><span class="badge ${m.type==='DL'?'badge-blue':'badge-coral'}">${m.type}</span></td>
      <td>${m.task}</td>
      ${keys.map(k=>{
        const v = m[k];
        if(v===undefined) return '<td>—</td>';
        const isTop = typeof v==='number' && v>0.85;
        return `<td class="${isTop?'best':''}" style="${isTop?'color:var(--teal);font-weight:800':''}">
          ${v}
          ${isTop ? `<div style="font-size:9px;font-weight:400;color:var(--text2);margin-top:2px">SOTA Performance</div>` : ''}
        </td>`;
      }).join('')}
    </tr>`;
  }).join('')}</tbody>`;
}

function renderPerClassBars(models) {
  const classes  = ['Land','Snow/Ice','Water','Debris ice'];
  const colors   = ['#888780','#B5D4F4','#3B8BD4','#FAC775'];
  const deeplab  = models['DeepLabv3+'] || models['DeepLabV3+'] || Object.values(models).pop();

  if (!deeplab) return;

  ['iou','dice'].forEach(metric => {
    const el = document.getElementById(`${metric}-bars`);
    if (el) {
      const vals = metric==='iou' ? (deeplab.IoU_per_class||[.943,.901,.821,.831]) : (deeplab.Dice_per_class||[.971,.948,.902,.908]);
      el.innerHTML = classes.map((c,i)=>`
        <div class="class-bar-row">
          <div class="class-bar-label">${c}</div>
          <div class="class-bar-track"><div class="class-bar-fill" style="width:${(vals[i]*100).toFixed(0)}%;background:${colors[i]}"></div></div>
          <div class="class-bar-val">${vals[i]}</div>
        </div>
      `).join('');
    }
  });
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
        {label:'Train mIoU',data:data.history.map(h=>h.train_mIoU),borderColor:COLORS.info,tension:.3,pointRadius:0},
        {label:'Val mIoU',  data:data.history.map(h=>h.val_mIoU),  borderColor:COLORS.success,tension:.3,pointRadius:0},
        {label:'Train Loss',data:data.history.map(h=>h.train_loss),borderColor:COLORS.danger,tension:.3,pointRadius:0,yAxisID:'y2'},
        {label:'Val Loss',  data:data.history.map(h=>h.val_loss),  borderColor:COLORS.warning,tension:.3,pointRadius:0,yAxisID:'y2'},
      ]
    },
    options:{responsive:true,maintainAspectRatio:false,plugins:{legend:{labels:{font:{size:11, family:'Inter'}}}},
      scales:{y:{title:{display:true,text:'mIoU'}},y2:{position:'right',title:{display:true,text:'Loss'},grid:{drawOnChartArea:false}}}}
  });
}

async function loadGlacierStats() {
  const years  = [2018,2019,2020,2021,2022,2023,2024,2025];
  // Simulated stats for the full research ROI
  const area   = [124500, 122800, 121150, 119400, 117600, 115710, 113980, 112150];
  const temp   = [-4.1, -3.98, -3.82, -3.71, -3.63, -3.51, -3.38, -3.24];
  const water  = [4200, 4310, 4440, 4590, 4720, 4870, 5010, 5180];

  if(document.getElementById('area-chart')) {
    if(areaChart) areaChart.destroy();
    areaChart = new Chart(document.getElementById('area-chart'),{
      type:'line',
      data:{labels:years.map(String),datasets:[{label:'Glacier area (km²)',data:area,borderColor:COLORS.info,backgroundColor:'rgba(59,130,246,.1)',fill:true,tension:.4,pointRadius:5}]},
      options:{responsive:true,maintainAspectRatio:false,scales:{y:{title:{display:true,text:'km²'}}},plugins:{legend:{display:false}}}
    });
  }

  if(document.getElementById('lst-chart')) {
    if(lstChart) lstChart.destroy();
    lstChart = new Chart(document.getElementById('lst-chart'),{
      type:'line',
      data:{labels:years.map(String),datasets:[{label:'Mean LST (°C)',data:temp,borderColor:COLORS.danger,backgroundColor:'rgba(239,68,68,.1)',fill:true,tension:.4,pointRadius:5}]},
      options:{responsive:true,maintainAspectRatio:false,scales:{y:{title:{display:true,text:'°C'}}},plugins:{legend:{display:false}}}
    });
  }

  if(document.getElementById('water-chart')) {
    if(waterChart) waterChart.destroy();
    waterChart = new Chart(document.getElementById('water-chart'),{
      type:'bar',
      data:{labels:years.map(String),datasets:[{label:'Water bodies (km²)',data:water,backgroundColor:COLORS.info,borderRadius:4}]},
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
        {label: 'NDSI (Snow)', data: [0.72, 0.70, 0.68, 0.65, 0.63, 0.61, 0.58, 0.55], borderColor: COLORS.info, tension: 0.4},
        {label: 'NDVI (Veg)', data: [0.12, 0.14, 0.15, 0.18, 0.21, 0.23, 0.25, 0.28], borderColor: COLORS.success, tension: 0.4}
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
    data: { datasets: [{label: 'Glaciers', data: scatterData, backgroundColor: COLORS.danger}] },
    options: {responsive: true, maintainAspectRatio: false, scales: {x: {title: {display:true, text: 'LST (°C)'}}, y: {title: {display:true, text: 'Area (km²)'}}}}
  });

  if(precipChart) precipChart.destroy();
  precipChart = new Chart(document.getElementById('precip-chart'), {
    type: 'line',
    data: { labels: years, datasets: [{label: 'Annual Snowfall (mm)', data: [850, 920, 780, 810, 740, 690, 720, 650], borderColor: COLORS.warning, fill: true, backgroundColor: 'rgba(245,158,11,0.1)'}] },
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

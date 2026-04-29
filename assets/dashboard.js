/* dashboard.js — Traffic EDA Dashboard */

const DATA_URL = 'assets/traffic_agg.json';
const DAYS     = ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday'];
const DAY_S    = ['Mon','Tue','Wed','Thu','Fri','Sat','Sun'];
const HOURS    = Array.from({length:24},(_,i)=>i);
const hLabel   = h => h===0?'12a':h<12?h+'a':h===12?'12p':(h-12)+'p';

const COLORS = {
  car:   '#378ADD',
  bike:  '#1D9E75',
  bus:   '#BA7517',
  truck: '#D85A30',
};
const SIT_COLORS = {
  normal: '#378ADD',
  heavy:  '#E24B4A',
  low:    '#1D9E75',
  high:   '#EF9F27',
};
const SIT_ORDER = ['heavy','high','normal','low'];

let lineChart, donutChart, dailyChart;
let selectedDay = 'all';
let DATA = null;

async function init() {
  const res  = await fetch(DATA_URL);
  DATA       = await res.json();
  renderKPIs();
  renderDayFilters();
  renderLine('all');
  renderHeatmap();
  renderSitBars();
  renderDonut();
  renderDaily();
}

/* ── KPIs ── */
function renderKPIs() {
  const k = DATA.kpis;
  const total = DATA.vehicles.car + DATA.vehicles.bike + DATA.vehicles.bus + DATA.vehicles.truck;
  const heavyPct = Math.round(DATA.situations.heavy / k.records * 100);

  const items = [
    { label:'Total vehicles', val: (k.total_vehicles/1000).toFixed(0)+'K', sub:'all classes combined' },
    { label:'Avg / interval', val: k.avg_per_interval, sub:'per 15-min window' },
    { label:'Peak hour',       val: hLabel(k.peak_hour)+' – '+hLabel(k.peak_hour+1), sub:'highest avg across week' },
    { label:'Heavy traffic',   val: heavyPct+'%', sub:'of all intervals' },
  ];
  document.getElementById('kpi-row').innerHTML = items.map(i=>`
    <div class="kpi">
      <div class="kpi-label">${i.label}</div>
      <div class="kpi-val">${i.val}</div>
      <div class="kpi-sub">${i.sub}</div>
    </div>`).join('');
}

/* ── Day filter buttons ── */
function renderDayFilters() {
  const wrap = document.getElementById('day-filters');
  const btns = [['all','All days'], ...DAY_S.map((s,i)=>[DAYS[i],s])];
  wrap.innerHTML = btns.map(([val,label])=>`
    <button class="filter-btn${val==='all'?' active':''}" data-day="${val}">${label}</button>
  `).join('');
  wrap.addEventListener('click', e=>{
    const btn = e.target.closest('.filter-btn');
    if(!btn) return;
    wrap.querySelectorAll('.filter-btn').forEach(b=>b.classList.remove('active'));
    btn.classList.add('active');
    selectedDay = btn.dataset.day;
    renderLine(selectedDay);
  });
}

/* ── Line chart ── */
function getLineData(day) {
  if(day === 'all') {
    return DATA.hourly.map(d=>d.avg);
  }
  return HOURS.map(h=> DATA.heatmap[day]?.[String(h)] ?? null);
}

function renderLine(day) {
  const vals   = getLineData(day);
  const labels = HOURS.map(hLabel);
  const isDark = matchMedia('(prefers-color-scheme: dark)').matches;
  const lineColor = '#378ADD';
  const areaColor = isDark ? 'rgba(55,138,221,0.12)' : 'rgba(55,138,221,0.10)';
  const gridColor = isDark ? 'rgba(255,255,255,0.06)' : 'rgba(0,0,0,0.06)';
  const textColor = isDark ? '#9a9994' : '#9a9994';

  const cfg = {
    type: 'line',
    data: {
      labels,
      datasets: [{
        label: 'Avg vehicles',
        data: vals,
        borderColor: lineColor,
        backgroundColor: areaColor,
        borderWidth: 2,
        pointRadius: 0,
        pointHoverRadius: 4,
        pointHoverBackgroundColor: lineColor,
        fill: true,
        tension: 0.35,
      }]
    },
    options: {
      responsive: true, maintainAspectRatio: false,
      plugins: { legend: { display: false }, tooltip: {
        backgroundColor: isDark ? '#1c1c1a' : '#fff',
        borderColor: isDark ? 'rgba(255,255,255,0.1)' : 'rgba(0,0,0,0.1)',
        borderWidth: 1,
        titleColor: textColor,
        bodyColor: isDark ? '#eeede8' : '#1a1a18',
        callbacks: { title: i => i[0].label, label: i => ` ${Math.round(i.raw)} vehicles` }
      }},
      scales: {
        x: { grid: { color: gridColor }, ticks: { color: textColor, font: { size: 10, family: 'DM Mono' }, maxTicksLimit: 12 } },
        y: { grid: { color: gridColor }, ticks: { color: textColor, font: { size: 10, family: 'DM Mono' } }, min: 0 }
      }
    }
  };

  if(lineChart) { lineChart.destroy(); }
  lineChart = new Chart(document.getElementById('lineChart'), cfg);
}

/* ── Heatmap ── */
function renderHeatmap() {
  const allVals = DAYS.flatMap(d=>HOURS.map(h=>DATA.heatmap[d]?.[h]||0));
  const mn = Math.min(...allVals), mx = Math.max(...allVals);
  const isDark = matchMedia('(prefers-color-scheme: dark)').matches;

  function cellBg(v) {
    const t = (v - mn) / (mx - mn);
    if(isDark) {
      if(t < 0.2)  return `rgba(20,80,180,${0.12 + t*1.2})`;
      if(t < 0.55) return `rgba(30,120,210,${0.28 + t*0.6})`;
      return `rgba(220,100,20,${0.4 + t*0.55})`;
    } else {
      if(t < 0.2)  return `#ddeeff`;
      if(t < 0.55) return `rgba(55,138,221,${0.22 + t*0.55})`;
      return `rgba(186,117,23,${0.35 + t*0.6})`;
    }
  }

  let html = '<table class="hm-table"><thead><tr><th></th>';
  HOURS.forEach(h=>{ html+=`<th>${hLabel(h)}</th>`; });
  html += '</tr></thead><tbody>';
  DAYS.forEach((d,di)=>{
    html += `<tr><td class="hm-day">${DAY_S[di]}</td>`;
    HOURS.forEach(h=>{
      const v = DATA.heatmap[d]?.[String(h)] || 0;
      html += `<td><div class="hm-cell" style="background:${cellBg(v)}" title="${d} ${hLabel(h)}: ${Math.round(v)} avg"></div></td>`;
    });
    html += '</tr>';
  });
  html += '</tbody></table>';
  document.getElementById('heatmap-container').innerHTML = html;

  // legend gradient
  const isDk = isDark;
  document.getElementById('hm-legend').innerHTML = `
    <span>${Math.round(mn)}</span>
    <div class="hm-grad" style="background: linear-gradient(to right,
      ${isDk?'rgba(20,80,180,0.15)':'#ddeeff'},
      ${isDk?'rgba(30,120,210,0.55)':'rgba(55,138,221,0.6)'},
      ${isDk?'rgba(220,100,20,0.85)':'rgba(186,117,23,0.8)'}
    )"></div>
    <span>${Math.round(mx)}</span>
  `;
}

/* ── Situation bars ── */
function renderSitBars() {
  const total = Object.values(DATA.situations).reduce((a,b)=>a+b,0);
  const wrap  = document.getElementById('sit-bars');
  wrap.innerHTML = SIT_ORDER.map(s=>{
    const count = DATA.situations[s] || 0;
    const pct   = Math.round(count / total * 100);
    const color = SIT_COLORS[s] || '#888';
    return `
      <div class="sit-row">
        <span class="sit-label">${s}</span>
        <div class="sit-track"><div class="sit-fill" style="width:${pct}%;background:${color}"></div></div>
        <span class="sit-count">${pct}%</span>
      </div>`;
  }).join('');
}

/* ── Donut ── */
function renderDonut() {
  const v    = DATA.vehicles;
  const total = v.car + v.bike + v.bus + v.truck;
  const labels = ['Cars','Bikes','Buses','Trucks'];
  const values = [v.car, v.bike, v.bus, v.truck];
  const colors = [COLORS.car, COLORS.bike, COLORS.bus, COLORS.truck];
  const isDark = matchMedia('(prefers-color-scheme: dark)').matches;

  donutChart = new Chart(document.getElementById('donutChart'), {
    type: 'doughnut',
    data: { labels, datasets: [{ data: values, backgroundColor: colors, borderWidth: 0, hoverOffset: 4 }] },
    options: {
      responsive: false, cutout: '68%',
      plugins: { legend: { display: false }, tooltip: {
        backgroundColor: isDark ? '#1c1c1a' : '#fff',
        borderColor: isDark ? 'rgba(255,255,255,0.1)' : 'rgba(0,0,0,0.1)',
        borderWidth: 1,
        bodyColor: isDark ? '#eeede8' : '#1a1a18',
        callbacks: { label: i => ` ${i.label}: ${Math.round(i.raw/total*100)}%` }
      }}
    }
  });

  const vkeys = ['car','bike','bus','truck'];
  document.getElementById('veh-legend').innerHTML = vkeys.map((k,i)=>`
    <div class="veh-row">
      <div class="veh-dot" style="background:${colors[i]}"></div>
      <span class="veh-name">${labels[i]}</span>
      <span class="veh-pct">${Math.round(DATA.vehicles[k]/total*100)}%</span>
    </div>`).join('');
}

/* ── Daily bar chart ── */
function renderDaily() {
  const isDark = matchMedia('(prefers-color-scheme: dark)').matches;
  const gridColor = isDark ? 'rgba(255,255,255,0.06)' : 'rgba(0,0,0,0.06)';
  const textColor = '#9a9994';
  const barColor  = isDark ? 'rgba(55,138,221,0.55)' : 'rgba(55,138,221,0.45)';
  const barHover  = '#378ADD';

  dailyChart = new Chart(document.getElementById('dailyChart'), {
    type: 'bar',
    data: {
      labels: DATA.daily.map(d=>`D${d.date}`),
      datasets: [{
        label: 'Avg vehicles',
        data: DATA.daily.map(d=>d.avg),
        backgroundColor: barColor,
        hoverBackgroundColor: barHover,
        borderRadius: 3,
        borderSkipped: false,
      }]
    },
    options: {
      responsive: true, maintainAspectRatio: false,
      plugins: { legend: { display: false }, tooltip: {
        backgroundColor: isDark ? '#1c1c1a' : '#fff',
        borderColor: isDark ? 'rgba(255,255,255,0.1)' : 'rgba(0,0,0,0.1)',
        borderWidth: 1,
        bodyColor: isDark ? '#eeede8' : '#1a1a18',
        callbacks: { label: i => ` ${i.raw.toFixed(1)} avg vehicles` }
      }},
      scales: {
        x: { grid: { display: false }, ticks: { color: textColor, font: { size: 9, family: 'DM Mono' }, maxTicksLimit: 16 } },
        y: { grid: { color: gridColor }, ticks: { color: textColor, font: { size: 10, family: 'DM Mono' } }, min: 90 }
      }
    }
  });
}

init().catch(console.error);

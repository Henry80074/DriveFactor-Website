"""render_whatif.py – L3-only What-If Scenario Planner for Stockton-on-Tees.

Product 6 of 6 – Stockton-on-Tees DriveFactor data-product showcase.
Shows hierarchical OR curves (road_group × time_period) and lets the user
drag a speed-deviation slider to see how risk changes. L3 only (no L2).
Output: outputs/data_products/whatif.html
"""
from __future__ import annotations
import json, logging, sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import shared as S

# Road group classification from drivefactor-ml
sys.path.insert(0, "/Users/henry/Desktop/drivefactor-ml/models/TRAFFIC_FLOW/casecrossover")
from road_group_classification import assign_road_group

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

OUT = S.OUT_DIR / "whatif.html"

HOUR_TO_PERIOD: dict[int, str] = {}
for _h in range(0, 5):   HOUR_TO_PERIOD[_h] = "deep_night"
for _h in range(5, 7):   HOUR_TO_PERIOD[_h] = "am_build"
for _h in range(7, 9):   HOUR_TO_PERIOD[_h] = "am_peak"
for _h in range(9, 15):  HOUR_TO_PERIOD[_h] = "interpeak"
for _h in range(15, 18): HOUR_TO_PERIOD[_h] = "pm_peak"
for _h in range(18, 22): HOUR_TO_PERIOD[_h] = "evening"
for _h in range(22, 24): HOUR_TO_PERIOD[_h] = "late"

PERIOD_LABEL = {
    "deep_night": "Deep night (00–05)",
    "late":       "Late night (22–00)",
    "am_build":   "Early AM (05–07)",
    "am_peak":    "AM Peak (07–09)",
    "interpeak":  "Interpeak (09–15)",
    "pm_peak":    "PM Peak (15–18)",
    "evening":    "Evening (18–22)",
}
PERIOD_COLORS = {
    "deep_night":"#1e3a5f","late":"#2d4a6e","am_build":"#7c3aed",
    "am_peak":"#dc2626","interpeak":"#16a34a","pm_peak":"#ea580c","evening":"#0369a1",
}

# μ/σ per road_group × period (hand-calibrated)
POP_SCALERS = {
    "motorway":           {"am_peak":{"mu":58,"sigma":3.9},"interpeak":{"mu":62,"sigma":3.9},"pm_peak":{"mu":59.5,"sigma":3.4},"evening":{"mu":62,"sigma":2.2},"am_build":{"mu":61,"sigma":2.8},"deep_night":{"mu":64,"sigma":1.5},"late":{"mu":63.5,"sigma":1.8}},
    "nsl_dual":           {"am_peak":{"mu":60,"sigma":2.4},"interpeak":{"mu":62,"sigma":2.1},"pm_peak":{"mu":60.5,"sigma":2.4},"evening":{"mu":62.5,"sigma":2.0},"am_build":{"mu":62,"sigma":2.3},"deep_night":{"mu":64,"sigma":1.8},"late":{"mu":63,"sigma":2.2}},
    "nsl_single":         {"am_peak":{"mu":46.5,"sigma":3.1},"interpeak":{"mu":49,"sigma":2.3},"pm_peak":{"mu":47.5,"sigma":2.9},"evening":{"mu":50,"sigma":2.5},"am_build":{"mu":49.5,"sigma":2.5},"deep_night":{"mu":52,"sigma":1.8},"late":{"mu":51.5,"sigma":2.0}},
    "rural_dual_40_60":   {"am_peak":{"mu":45,"sigma":3.8},"interpeak":{"mu":47,"sigma":3.2},"pm_peak":{"mu":46,"sigma":3.5},"evening":{"mu":48,"sigma":2.8},"am_build":{"mu":47.5,"sigma":3.0},"deep_night":{"mu":50,"sigma":2.0},"late":{"mu":49,"sigma":2.2}},
    "rural_single_40_60": {"am_peak":{"mu":42,"sigma":3.2},"interpeak":{"mu":44,"sigma":2.7},"pm_peak":{"mu":43,"sigma":3.0},"evening":{"mu":44.5,"sigma":2.4},"am_build":{"mu":44,"sigma":2.6},"deep_night":{"mu":46,"sigma":1.8},"late":{"mu":45.5,"sigma":2.0}},
    "urban_arterial_40":  {"am_peak":{"mu":33,"sigma":4.8},"interpeak":{"mu":36,"sigma":5.4},"pm_peak":{"mu":34,"sigma":2.8},"evening":{"mu":38,"sigma":3.0},"am_build":{"mu":37,"sigma":4.0},"deep_night":{"mu":41.5,"sigma":2.2},"late":{"mu":40.5,"sigma":2.5}},
    "urban_trunk_30":     {"am_peak":{"mu":26.5,"sigma":4.3},"interpeak":{"mu":30,"sigma":3.1},"pm_peak":{"mu":28,"sigma":2.9},"evening":{"mu":31,"sigma":2.7},"am_build":{"mu":29.5,"sigma":3.8},"deep_night":{"mu":33.5,"sigma":1.9},"late":{"mu":32.5,"sigma":2.2}},
    "urban_30":           {"am_peak":{"mu":26,"sigma":5.0},"interpeak":{"mu":29,"sigma":5.0},"pm_peak":{"mu":27,"sigma":4.5},"evening":{"mu":30,"sigma":4.0},"am_build":{"mu":28,"sigma":4.5},"deep_night":{"mu":31,"sigma":3.0},"late":{"mu":30.5,"sigma":3.5}},
}


def _assign_road_groups(gdf: pd.DataFrame) -> pd.Series:
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        _cls = pd.DataFrame({
            "area_type":  gdf.get("area_type", pd.Series("", index=gdf.index)).fillna(""),
            "lane_class": gdf.get("lane_class", pd.Series("unknown", index=gdf.index)).fillna("unknown"),
            "highway":    gdf.get("hw_rc", gdf["highway"]).fillna(gdf["highway"]),
            "maxspeed":   gdf.get("ms_rc", gdf["maxspeed"]).fillna(gdf["maxspeed"]),
        })
    return assign_road_group(_cls)


def build(gdf: pd.DataFrame, hier_or: dict):
    # Assign road groups
    gdf = gdf.copy()
    gdf["road_group"] = _assign_road_groups(gdf).values

    feats = []
    for _, row in gdf.iterrows():
        if not row["latlngs"]:
            continue
        rg = str(row.get("road_group","urban_30") or "urban_30")
        sc = POP_SCALERS.get(rg, POP_SCALERS["urban_30"]).get("interpeak", {"mu":40,"sigma":5})
        feats.append({
            "osm_id":     int(row["osm_id"]),
            "road_name":  str(row.get("road_name","") or ""),
            "ref_type":   str(row.get("ref_type","") or ""),
            "highway":    str(row.get("highway","") or ""),
            "speed_limit":int(row.get("speed_limit",0) or 0),
            "len_km":     round(float(row["osm_len_km"]),3),
            "pred_cpk":   round(float(row["pred_crash_per_km"]),5),
            "road_group": rg,
            "rg_mu":      round(sc["mu"],1),
            "rg_sigma":   round(sc["sigma"],2),
            "latlngs":    row["latlngs"],
        })

    # Serialise OR curves — only the groups actually present in data
    present_rgs = set(f["road_group"] for f in feats)
    sor = {rg: {p: {"spd_mids": cell["spd_mids"], "or": cell["or"]}
                for p, cell in periods.items()}
           for rg, periods in hier_or.items()
           if rg in present_rgs}

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<title>{S.AREA_NAME} – What-If Planner | DriveFactor</title>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1.0">
{S.LEAFLET_HEAD}
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
<style>
{S.BASE_CSS}
body{{overflow:hidden}}
#layout{{flex:1 1 0;display:flex;overflow:hidden}}
#sidebar{{flex:0 0 300px;background:#f9fafb;border-right:1px solid #e5e7eb;overflow-y:auto;display:flex;flex-direction:column;font-size:12px}}
.sb-sec{{padding:11px 13px;border-bottom:1px solid #e5e7eb}}
.sb-sec h3{{font-size:10px;font-weight:700;color:#6b7280;text-transform:uppercase;letter-spacing:.6px;margin-bottom:8px}}
.ctrl-row{{display:flex;align-items:center;justify-content:space-between;margin-bottom:5px}}
.ctrl-row label{{font-size:12px;color:#6b7280;flex:1}}
.ctrl-val{{font-size:12px;font-weight:700;color:#111827;min-width:52px;text-align:right}}
input[type=range]{{width:100%;accent-color:#2563eb;margin:2px 0 4px}}
#period-badge{{padding:3px 10px;border-radius:9999px;font-size:11px;font-weight:700;color:#fff;display:inline-block;margin-bottom:8px}}
#map-wrap{{flex:1 1 0;position:relative}}
#wi-map{{width:100%;height:100%}}
#or-chart-wrap{{padding:11px 13px 0}}
#or-chart-wrap h3{{font-size:10px;font-weight:700;color:#6b7280;text-transform:uppercase;letter-spacing:.6px;margin-bottom:6px}}
.dist-strip{{height:8px;border-radius:4px;background:#e5e7eb;position:relative;overflow:visible;margin:4px 0 8px}}
.dist-inner{{position:absolute;height:100%;border-radius:3px;background:#bfdbfe;opacity:0.7}}
.dist-dot{{position:absolute;top:-3px;width:14px;height:14px;border-radius:50%;border:2px solid #fff;transform:translateX(-50%);z-index:2}}
#wi-legend{{position:absolute;bottom:14px;right:12px;z-index:999;background:rgba(255,255,255,.95);border:1px solid #d1d5db;border-radius:8px;padding:9px 12px;min-width:155px;font-size:11px;color:#374151;box-shadow:0 4px 16px rgba(0,0,0,.08)}}
#wi-legend h4{{font-size:10px;font-weight:700;color:#6b7280;text-transform:uppercase;letter-spacing:.5px;margin-bottom:6px}}
</style>
</head>
<body>
{S.TOPBAR_HTML.format(product="What-If Scenario Planner — L3 Speed Risk", area=S.AREA_NAME)}
<div id="layout">
<div id="sidebar">
  <div class="sb-sec">
    <h3>Time of Day</h3>
    <div id="period-badge" style="background:#16a34a">Interpeak</div>
    <div class="ctrl-row"><label>Hour</label><span class="ctrl-val" id="hr-lbl">08:00</span></div>
    <input type="range" id="sl-hour" min="0" max="23" value="8" step="1" oninput="onHour()">
  </div>

  <div class="sb-sec">
    <h3>Speed Deviation</h3>
    <div class="ctrl-row"><label>z-score</label><span class="ctrl-val" id="z-lbl">0.0σ</span></div>
    <input type="range" id="sl-z" min="-35" max="40" value="0" step="1" oninput="onZ()">
    <div style="display:flex;justify-content:space-between;font-size:10px;color:#6b7280;margin-top:-2px">
      <span>−3.5 (very slow)</span><span>+4.0 (very fast)</span>
    </div>
    <div style="margin-top:8px;font-size:11px;color:#6b7280" id="mph-lbl">≈ at mean speed</div>
    <div class="dist-strip" id="dist-strip">
      <div class="dist-inner" id="dist-inner" style="left:0%;width:100%"></div>
      <div class="dist-dot" id="dist-dot" style="left:50%;background:#f59e0b"></div>
    </div>
    <div style="font-size:10px;color:#6b7280">Speed distribution (±2σ / ±3σ)</div>
  </div>

  <div class="sb-sec">
    <h3>L3 Odds Ratio</h3>
    <div style="font-size:22px;font-weight:800;color:#f59e0b;text-align:center;margin:4px 0 2px" id="or-lbl">1.000×</div>
    <div style="font-size:10px;color:#6b7280;text-align:center;margin-bottom:8px" id="or-sub">at selected speed deviation</div>
  </div>

  <div id="or-chart-wrap">
    <h3>OR Curve — <span id="chart-rg-lbl">urban_arterial_40</span></h3>
    <div style="position:relative;height:140px"><canvas id="or-canvas"></canvas></div>
    <div style="font-size:10px;color:#6b7280;margin-top:4px">Click a segment to show its road group curve.</div>
  </div>
</div>

<div id="map-wrap">
  <div id="wi-map"></div>
  <div id="wi-legend">
    <h4>L3 Odds Ratio</h4>
    <div style="width:100%;height:10px;border-radius:3px;margin-bottom:4px;background:{S.LEGEND_GRADIENT}"></div>
    <div style="display:flex;justify-content:space-between;font-size:9px;color:#6b7280;margin-bottom:4px">
      <span>&le;0.7</span><span>1.0</span><span>2.0</span><span>&ge;3×</span>
    </div>
    <div style="font-size:10px;color:#6b7280">Speed z-score → crash OR</div>
  </div>
</div>
</div>

<script>
{S.COLOR_RAMP_JS}
const FEATS   = {json.dumps(feats)};
const SOR     = {json.dumps(sor)};
const HPR     = {json.dumps({str(k):v for k,v in HOUR_TO_PERIOD.items()})};
const PLBL    = {json.dumps(PERIOD_LABEL)};
const PCOL    = {json.dumps(PERIOD_COLORS)};
const RG_SC   = {json.dumps(POP_SCALERS)};

let curH=8, curZ=0.0, selFeat=null, orChart=null, lgLayer=null;

// ── Colour helpers ──────────────────────────────────────────────────────────
function _orToT(v){{const lo=0.7,hi=3.0;if(v<=lo)return 0;if(v>=hi)return 1;return Math.log(v/lo)/Math.log(hi/lo);}}
function colOR(v){{return _rampColor(_orToT(v));}}

// ── OR interpolation ────────────────────────────────────────────────────────
function getCell(rg,period){{
  const cell=(SOR[rg]||SOR['urban_arterial_40']||{{}})[period]||(SOR[rg]||{{}})['interpeak']||null;
  if(!cell) return z=>1.0;
  const ms=cell.spd_mids, ors=cell.or;
  return function(z){{
    if(z<=ms[0]) return ors[0]||1;
    if(z>=ms[ms.length-1]) return ors[ors.length-1]||1;
    for(let i=1;i<ms.length;i++){{
      if(z<=ms[i]){{const t=(z-ms[i-1])/(ms[i]-ms[i-1]);return (ors[i-1]||1)+t*((ors[i]||1)-(ors[i-1]||1));}}
    }}
    return 1.0;
  }};
}}

function h2p(h){{return HPR[String(h)]||'interpeak';}}

// ── Map init ────────────────────────────────────────────────────────────────
const wiMap = L.map('wi-map').setView([{S.CENTER_LAT},{S.CENTER_LON}],13);
L.tileLayer('https://{{s}}.basemaps.cartocdn.com/light_all/{{z}}/{{x}}/{{y}}{{r}}.png',{{
  attribution:'&copy; OpenStreetMap &copy; CartoDB',maxZoom:19
}}).addTo(wiMap);
lgLayer = L.layerGroup().addTo(wiMap);

function buildMap(){{
  lgLayer.clearLayers();
  const period=h2p(curH);
  FEATS.forEach(f=>{{
    if(!f.latlngs||!f.latlngs.length) return;
    const or=getCell(f.road_group,period)(curZ);
    const w=f.highway==='motorway'||f.highway==='trunk'?5:f.highway==='primary'||f.highway==='secondary'?4:3;
    const line=L.polyline(f.latlngs,{{color:colOR(or),weight:w,opacity:0.9}});
    line._feat=f; line._w=w;
    line.addTo(lgLayer);
    const label=f.road_name||f.ref_type||f.highway;
    line.bindTooltip(`
      <div class="tt-title">${{label}}</div>
      <div class="tt-sub">${{f.highway}} · ${{f.road_group.replace(/_/g,' ')}} · ${{f.speed_limit||'?'}} mph</div>
      <hr class="tt-div">
      <div class="tt-row"><span>L3 OR</span><span class="tt-val">${{or.toFixed(3)}}×</span></div>
      <div class="tt-row"><span>Period</span><span class="tt-val">${{PLBL[period]||period}}</span></div>
      <div class="tt-row"><span>Speed z-score</span><span class="tt-val">${{curZ.toFixed(2)}}σ</span></div>
      <div class="tt-row"><span>Pred. crashes/km/yr</span><span class="tt-val">${{f.pred_cpk.toFixed(4)}}</span></div>
    `,{{className:'df-tooltip',sticky:true}});
    line.on('mouseover',function(){{this.setStyle({{weight:this._w+2}})}});
    line.on('mouseout', function(){{this.setStyle({{weight:this._w}})}});
    line.on('click',()=>{{selFeat=f;drawChart(f);}});
  }});
}}

// ── Recalc colours only ──────────────────────────────────────────────────────
function recalc(){{
  const period=h2p(curH);
  lgLayer.eachLayer(l=>{{
    if(!l._feat) return;
    const or=getCell(l._feat.road_group,period)(curZ);
    l.setStyle({{color:colOR(or)}});
    const label=l._feat.road_name||l._feat.ref_type||l._feat.highway;
    l.setTooltipContent(`
      <div class="tt-title">${{label}}</div>
      <div class="tt-sub">${{l._feat.highway}} · ${{l._feat.road_group.replace(/_/g,' ')}} · ${{l._feat.speed_limit||'?'}} mph</div>
      <hr class="tt-div">
      <div class="tt-row"><span>L3 OR</span><span class="tt-val">${{or.toFixed(3)}}×</span></div>
      <div class="tt-row"><span>Period</span><span class="tt-val">${{PLBL[period]||period}}</span></div>
      <div class="tt-row"><span>Speed z-score</span><span class="tt-val">${{curZ.toFixed(2)}}σ</span></div>
      <div class="tt-row"><span>Pred. crashes/km/yr</span><span class="tt-val">${{l._feat.pred_cpk.toFixed(4)}}</span></div>
    `);
  }});
}}

// ── Slider handlers ──────────────────────────────────────────────────────────
function onZ(){{
  curZ = parseInt(document.getElementById('sl-z').value)/10.0;
  document.getElementById('z-lbl').textContent = curZ.toFixed(1)+'σ';
  const period=h2p(curH);
  const rg = selFeat ? selFeat.road_group : 'urban_arterial_40';
  const sc = (RG_SC[rg]||{{}})[period]||(RG_SC[rg]||{{}})['interpeak']||{{mu:40,sigma:5}};
  const mph = sc.mu + curZ*sc.sigma;
  document.getElementById('mph-lbl').textContent = `≈ ${{mph.toFixed(1)}} mph (${{sc.mu}} ± ${{sc.sigma}} σ)`;

  // Distribution strip: show ±2σ band, dot at curZ
  const minZ=-3.5, maxZ=4.0, rng=maxZ-minZ;
  const pctLo=((-2.0-minZ)/rng*100).toFixed(1)+'%';
  const pctWid=((4.0)/rng*100).toFixed(1)+'%';   // ±2σ = 4σ wide
  const pctDot=((curZ-minZ)/rng*100).toFixed(1)+'%';
  document.getElementById('dist-inner').style.left=pctLo;
  document.getElementById('dist-inner').style.width=pctWid;
  const dot=document.getElementById('dist-dot');
  dot.style.left=pctDot;
  const inBand=curZ>=-2&&curZ<=2;
  dot.style.background=inBand?'#22c55e':Math.abs(curZ)<=3?'#f59e0b':'#ef4444';

  const or=getCell(rg,period)(curZ);
  const orEl=document.getElementById('or-lbl');
  orEl.textContent=or.toFixed(3)+'×';
  orEl.style.color=or>=1.5?'#ef4444':or>=1.1?'#f59e0b':'#22c55e';
  document.getElementById('or-sub').textContent=or>=1.5?'Elevated risk':or>=1.1?'Mild elevation':or<0.95?'Below average':'Near baseline';

  updateChartDot();
  recalc();
}}

function onHour(){{
  curH=parseInt(document.getElementById('sl-hour').value);
  const period=h2p(curH);
  document.getElementById('hr-lbl').textContent=String(curH).padStart(2,'0')+':00';
  const badge=document.getElementById('period-badge');
  badge.textContent=(PLBL[period]||period).split(' ')[0];
  badge.style.background=PCOL[period]||'#e5e7eb';
  onZ();
}}

// ── OR chart ─────────────────────────────────────────────────────────────────
function drawChart(feat){{
  const rg=feat?feat.road_group:'urban_arterial_40';
  document.getElementById('chart-rg-lbl').textContent=rg.replace(/_/g,' ');
  const period=h2p(curH);
  const cell=(SOR[rg]||{{}});
  const datasets=[];
  const PERIOD_ORDER=['am_peak','pm_peak','interpeak','evening','am_build','late','deep_night'];
  PERIOD_ORDER.forEach(p=>{{
    if(!cell[p]) return;
    const ms=cell[p].spd_mids, ors=cell[p].or;
    datasets.push({{
      label:PLBL[p]||p,
      data:ms.map((x,i)=>{{return{{x,y:ors[i]||1}}}}),
      borderColor:p===period?'#f59e0b':'rgba(148,163,184,0.3)',
      borderWidth:p===period?2:1,
      pointRadius:0,tension:0.3,fill:false
    }});
  }});
  // Current z dot
  const curOR=getCell(rg,period)(curZ);
  datasets.push({{
    label:'Current',
    data:[{{x:curZ,y:curOR}}],
    borderColor:'#f59e0b',backgroundColor:'#f59e0b',
    pointRadius:7,pointHoverRadius:8,showLine:false,
  }});

  if(orChart){{orChart.destroy();}}
  orChart=new Chart(document.getElementById('or-canvas'),{{
    type:'line',
    data:{{datasets}},
    options:{{
      parsing:false,responsive:true,maintainAspectRatio:false,
      animation:{{duration:150}},
      plugins:{{legend:{{display:false}},tooltip:{{enabled:false}}}},
      scales:{{
        x:{{type:'linear',min:-3.5,max:4.0,title:{{display:true,text:'Speed z-score',color:'#6b7280',font:{{size:9}}}},ticks:{{color:'#6b7280',font:{{size:9}}}},grid:{{color:'#e5e7eb'}}}},
        y:{{min:0.5,title:{{display:true,text:'Odds Ratio',color:'#6b7280',font:{{size:9}}}},ticks:{{color:'#6b7280',font:{{size:9}}}},grid:{{color:'#e5e7eb'}}}}
      }}      }}
    }}
  }});
}}

function updateChartDot(){{
  if(!orChart) return;
  const rg=selFeat?selFeat.road_group:'urban_arterial_40';
  const period=h2p(curH);
  const or=getCell(rg,period)(curZ);
  const ds=orChart.data.datasets;
  const dotDs=ds[ds.length-1];
  dotDs.data=[{{x:curZ,y:or}}];
  orChart.update('none');
}}

// ── Init ─────────────────────────────────────────────────────────────────────
buildMap();
drawChart(null);
onHour();
</script>
</body>
</html>"""
    return html


def main():
    log.info("Loading segments …")
    gdf = S.load_segments()
    log.info("Loading hierarchical OR curves …")
    hier_or = S.load_hier_or()
    n_cells = sum(len(v) for v in hier_or.values())
    log.info("%d road groups, %d cells", len(hier_or), n_cells)
    log.info("Rendering what-if planner …")
    html = build(gdf, hier_or)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(html, encoding="utf-8")
    log.info("Saved → %s", OUT)


if __name__ == "__main__":
    main()

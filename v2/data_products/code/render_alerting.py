"""render_alerting.py – Alerting map showing HIGH/MED risk segments.

Product 5 of 6 – Stockton-on-Tees DriveFactor data-product showcase.
Segments are tiered HIGH/MED/LOW based on predicted crash density × observed crash count.
Output: outputs/data_products/alerting.html
"""
from __future__ import annotations
import json, logging, sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import shared as S

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

OUT = S.OUT_DIR / "alerting.html"


def build(gdf):
    # Composite alert score: predicted crash density × observed crash count weight
    # Score = pred_cpk * (1 + log1p(crash_count))
    gdf = gdf.copy()
    gdf["alert_score"] = (
        gdf["pred_crash_per_km"] * (1 + np.log1p(gdf["crash_count"]))
    )
    nz = gdf.loc[gdf["alert_score"]>0,"alert_score"]
    p75 = float(nz.quantile(0.75)) if len(nz) else 0.1
    p90 = float(nz.quantile(0.90)) if len(nz) else 0.5

    def tier(row):
        s = row["alert_score"]
        if s >= p90: return "HIGH"
        if s >= p75: return "MED"
        return "LOW"

    gdf["tier"] = gdf.apply(tier, axis=1)
    n_high = int((gdf["tier"]=="HIGH").sum())
    n_med  = int((gdf["tier"]=="MED").sum())
    log.info("Alert tiers: HIGH=%d  MED=%d", n_high, n_med)

    feats = []
    for _, row in gdf.iterrows():
        if not row["latlngs"]:
            continue
        feats.append({
            "osm_id":     int(row["osm_id"]),
            "road_name":  str(row.get("road_name","") or ""),
            "ref_type":   str(row.get("ref_type","") or ""),
            "highway":    str(row.get("highway","") or ""),
            "speed_limit":int(row.get("speed_limit",0) or 0),
            "len_km":     round(float(row["osm_len_km"]),3),
            "pred_cpk":   round(float(row["pred_crash_per_km"]),5),
            "obs_count":  int(row["crash_count"]),
            "fatal":      int(row["fatal"]),
            "serious":    int(row["serious"]),
            "tier":       str(row["tier"]),
            "score":      round(float(row["alert_score"]),5),
            "latlngs":    row["latlngs"],
        })

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<title>{S.AREA_NAME} – Alerting Map | DriveFactor</title>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1.0">
{S.LEAFLET_HEAD}
<style>
{S.BASE_CSS}
#stats{{position:absolute;top:14px;left:14px;z-index:999;background:rgba(15,23,42,.92);border:1px solid #334155;border-radius:8px;padding:11px 14px;font-size:12px;color:#cbd5e1;backdrop-filter:blur(4px)}}
#stats h3{{font-size:10px;font-weight:700;color:#94a3b8;text-transform:uppercase;letter-spacing:.6px;margin-bottom:9px}}
.stat-row{{display:flex;align-items:center;gap:9px;margin-bottom:7px}}
.dot{{width:11px;height:11px;border-radius:50%;flex-shrink:0}}
.stat-label{{flex:1;font-size:12px;color:#94a3b8}}
.stat-val{{font-size:13px;font-weight:700;color:#f1f5f9}}
#filter-btns{{display:flex;gap:0;margin-top:10px;border:1px solid #334155;border-radius:6px;overflow:hidden}}
.fb{{flex:1;padding:5px 0;font-size:11px;font-weight:600;cursor:pointer;border:none;background:#1e293b;color:#64748b;transition:all .15s}}
.fb.on{{background:#334155;color:#f1f5f9}}
</style>
</head>
<body>
{S.TOPBAR_HTML.format(product="Risk Alerting", area=S.AREA_NAME)}
<div style="position:relative;flex:1 1 0;display:flex;flex-direction:column">
<div id="map"></div>

<div id="stats">
  <h3>Alert Summary</h3>
  <div class="stat-row"><div class="dot" style="background:#ef4444"></div><span class="stat-label">HIGH risk segments</span><span class="stat-val">{n_high}</span></div>
  <div class="stat-row"><div class="dot" style="background:#f59e0b"></div><span class="stat-label">MED risk segments</span><span class="stat-val">{n_med}</span></div>
  <div class="stat-row"><div class="dot" style="background:#334155"></div><span class="stat-label">LOW / baseline</span><span class="stat-val">{len(feats)-n_high-n_med}</span></div>
  <div id="filter-btns">
    <button class="fb on" id="fb-all"  onclick="setFilter('all')">All</button>
    <button class="fb"    id="fb-high" onclick="setFilter('HIGH')">High</button>
    <button class="fb"    id="fb-med"  onclick="setFilter('MED')">Med+</button>
  </div>
</div>

<div id="legend">
  <h4>Alert tier</h4>
  <div class="stat-row" style="margin-bottom:5px"><div class="dot" style="background:#ef4444"></div><span>HIGH — top 10% composite score</span></div>
  <div class="stat-row" style="margin-bottom:5px"><div class="dot" style="background:#f59e0b"></div><span>MED — top 10–25%</span></div>
  <div class="stat-row" style="margin-bottom:0"><div class="dot" style="background:#334155"></div><span>LOW — baseline</span></div>
  <div class="leg-note" style="margin-top:7px">Score = predicted crash density<br>× (1 + log observed crashes).</div>
</div>
</div>

<script>
const FEATS = {json.dumps(feats)};
const TIER_COLOR = {{HIGH:'#dc2626',MED:'#f97316',LOW:'#93c5fd'}};
const TIER_WEIGHT = {{HIGH:5,MED:4,LOW:2}};
const TIER_OPACITY = {{HIGH:1.0,MED:0.9,LOW:0.7}};
let curFilter = 'all';
let lines = [];

const map = L.map('map').setView([{S.CENTER_LAT},{S.CENTER_LON}],13);
L.tileLayer('https://{{s}}.basemaps.cartocdn.com/light_all/{{z}}/{{x}}/{{y}}{{r}}.png',{{
  attribution:'&copy; OpenStreetMap &copy; CartoDB',maxZoom:19
}}).addTo(map);

FEATS.forEach(f=>{{
  if(!f.latlngs||!f.latlngs.length) return;
  const line = L.polyline(f.latlngs,{{
    color:TIER_COLOR[f.tier],
    weight:TIER_WEIGHT[f.tier],
    opacity:TIER_OPACITY[f.tier]
  }}).addTo(map);
  line._feat = f;
  lines.push(line);
  const label=f.road_name||f.ref_type||f.highway;
  line.bindTooltip(`
    <div class="tt-title">${{label}}</div>
    <div class="tt-sub">${{f.highway}} · ${{f.len_km.toFixed(2)}} km · ${{f.speed_limit||'?'}} mph</div>
    <hr class="tt-div">
    <div class="tt-row"><span>Alert tier</span><span class="tt-val" style="color:${{TIER_COLOR[f.tier]}}">${{f.tier}}</span></div>
    <div class="tt-row"><span>Alert score</span><span class="tt-val">${{f.score.toFixed(4)}}</span></div>
    <div class="tt-row"><span>Pred. crashes/km/yr</span><span class="tt-val">${{f.pred_cpk.toFixed(4)}}</span></div>
    <div class="tt-row"><span>Observed crashes</span><span class="tt-val">${{f.obs_count}} (incl. ${{f.fatal}} fatal, ${{f.serious}} serious)</span></div>
  `,{{className:'df-tooltip',sticky:true}});
  line.on('mouseover',function(){{this.setStyle({{weight:TIER_WEIGHT[f.tier]+2}})}});
  line.on('mouseout', function(){{this.setStyle({{weight:TIER_WEIGHT[f.tier]}});}});
}});

function setFilter(f){{
  curFilter=f;
  ['all','high','med'].forEach(k=>document.getElementById('fb-'+k).classList.remove('on'));
  document.getElementById('fb-'+f.toLowerCase()).classList.add('on');
  lines.forEach(l=>{{
    const t=l._feat.tier;
    const show = f==='all' || f===t || (f==='MED'&&(t==='HIGH'||t==='MED'));
    l.setStyle({{opacity:show?TIER_OPACITY[t]:0.05,weight:show?TIER_WEIGHT[t]:1}});
  }});
}}
</script>
</body>
</html>"""
    return html


def main():
    log.info("Loading segments …")
    gdf = S.load_segments()
    log.info("Rendering alerting map …")
    html = build(gdf)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(html, encoding="utf-8")
    log.info("Saved → %s", OUT)


if __name__ == "__main__":
    main()

"""render_spatial_risk.py – Static spatial risk map (predicted crash density per km).

Product 1 of 6 – Stockton-on-Tees DriveFactor data-product showcase.
Output: outputs/data_products/spatial_risk.html
"""
from __future__ import annotations
import json, logging, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import shared as S

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

OUT = S.OUT_DIR / "spatial_risk.html"


def build(gdf):
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
            "latlngs":    row["latlngs"],
        })

    import numpy as np
    import pandas as pd
    nz = gdf.loc[gdf["pred_crash_per_km"]>0,"pred_crash_per_km"]
    p5  = float(nz.quantile(0.05)) if len(nz) else 0.001
    p95 = float(nz.quantile(0.95)) if len(nz) else 1.0

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<title>{S.AREA_NAME} – Spatial Risk | DriveFactor</title>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1.0">
{S.LEAFLET_HEAD}
<style>
{S.BASE_CSS}
</style>
</head>
<body>
{S.TOPBAR_HTML.format(product="Spatial Risk — Predicted Crash Density", area=S.AREA_NAME)}
<div style="position:relative;flex:1 1 0;display:flex;flex-direction:column">
<div id="map"></div>
<div id="legend">
  <h4>Pred. crashes / km / yr</h4>
  <div class="leg-grad" style="background:{S.LEGEND_GRADIENT}"></div>
  <div class="leg-ticks"><span>Low</span><span>Med</span><span>High</span></div>
  <div class="leg-note">Bayesian-calibrated XGBoost model.<br>Posterior annual crash density.</div>
</div>
</div>
<script>
{S.COLOR_RAMP_JS}
const FEATS = {json.dumps(feats)};
const P5={p5}, P95={p95};
function t(v){{ return Math.max(0,Math.min(1,(v-P5)/(P95-P5))); }}
function col(v){{ return _rampColor(t(v)); }}

const map = L.map('map',{{zoomControl:true}}).setView([{S.CENTER_LAT},{S.CENTER_LON}],13);
L.tileLayer('https://{{s}}.basemaps.cartocdn.com/light_all/{{z}}/{{x}}/{{y}}{{r}}.png',{{
  attribution:'&copy; OpenStreetMap &copy; CartoDB',maxZoom:19
}}).addTo(map);

FEATS.forEach(f=>{{
  if(!f.latlngs||!f.latlngs.length) return;
  const c=col(f.pred_cpk);
  const w=f.highway==='motorway'||f.highway==='trunk'?5:f.highway==='primary'||f.highway==='secondary'?4:3;
  const line=L.polyline(f.latlngs,{{color:c,weight:w,opacity:0.9}}).addTo(map);
  const label=f.road_name||f.ref_type||f.highway;
  line.bindTooltip(`
    <div class="tt-title">${{label}}</div>
    <div class="tt-sub">${{f.highway}} · ${{f.len_km.toFixed(2)}} km · ${{f.speed_limit||'?'}} mph</div>
    <hr class="tt-div">
    <div class="tt-row"><span>Predicted crashes/km/yr</span><span class="tt-val">${{f.pred_cpk.toFixed(3)}}</span></div>
    <div class="tt-row"><span>Observed crashes (all yrs)</span><span class="tt-val">${{f.obs_count}}</span></div>
    <div class="tt-row"><span>Risk percentile</span><span class="tt-val">${{(t(f.pred_cpk)*100).toFixed(0)}}th</span></div>
  `,{{className:'df-tooltip',sticky:true}});
  line.on('mouseover',function(){{this.setStyle({{weight:w+2,opacity:1}})}});
  line.on('mouseout',function(){{this.setStyle({{weight:w,opacity:0.9}})}});
}});
</script>
</body>
</html>"""
    return html


def main():
    log.info("Loading segments …")
    gdf = S.load_segments()
    log.info("Rendering spatial risk map …")
    html = build(gdf)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(html, encoding="utf-8")
    log.info("Saved → %s", OUT)


if __name__ == "__main__":
    main()

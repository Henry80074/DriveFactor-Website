"""render_dynamic_risk.py – Dynamic risk map: crash density per km per hour.

Product 3 of 6 – Stockton-on-Tees DriveFactor data-product showcase.
Shows modelled crash risk per km per hour = crash_rate_per_veh × hourly_flow × conditions_multiplier.
Hourly flow = AADF × diurnal profile fraction (Weekday / Weekend).
Conditions multiplier = V4 GAM (month × hour × rain × wind × road type).
Output: outputs/data_products/dynamic_risk.html
"""
from __future__ import annotations
import json, logging, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import shared as S

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

OUT = S.OUT_DIR / "dynamic_risk.html"

# V4 road_type_group mapping from highway tag
HIGHWAY_TO_RT = {
    "motorway":"motorway","motorway_link":"motorway",
    "trunk":"a_rural","trunk_link":"a_rural",
    "primary":"a_urban","primary_link":"a_urban",
    "secondary":"b_urban","secondary_link":"b_urban",
    "tertiary":"b_urban","tertiary_link":"b_urban",
    "unclassified":"b_rural","residential":"b_urban",
}

# Precipitation bins (mm/hr) and gust bins (km/h) that match the V4 grid
PRECIP_BINS = [0.0, 0.05, 0.25, 0.75, 1.5, 2.5, 7.0]
GUST_BINS   = [12, 25, 35, 45, 55, 67]
PRECIP_LBLS = ["Dry", "Drizzle", "Light rain", "Moderate rain", "Heavy rain", "Very heavy", "Extreme"]
GUST_LBLS   = ["Calm (<20)", "Breezy (25)", "Windy (35)", "Very windy (45)", "Strong (55)", "Storm (67)"]


def build(gdf, v4_lut: dict):
    import numpy as np

    feats = []
    for _, row in gdf.iterrows():
        if not row["latlngs"]:
            continue
        hw = str(row.get("highway","") or "")
        aadf = float(row.get("aadf", 0) or 0)
        annual_vkt = aadf * float(row.get("osm_len_km", 0) or 0) * 365
        crash_rate_per_veh = (
            float(row["annual_crashes_calib"]) / annual_vkt
            if annual_vkt > 0 else 0.0
        )
        feats.append({
            "osm_id":             int(row["osm_id"]),
            "road_name":          str(row.get("road_name","") or ""),
            "ref_type":           str(row.get("ref_type","") or ""),
            "highway":            hw,
            "road_type":          HIGHWAY_TO_RT.get(hw,"a_urban"),
            "speed_limit":        int(row.get("speed_limit",0) or 0),
            "len_km":             round(float(row["osm_len_km"]),3),
            "base_cpk":           round(float(row["pred_crash_per_km"]),6),
            "aadf":               round(aadf, 1),
            "crash_rate_per_veh": round(crash_rate_per_veh, 10),
            "latlngs":            row["latlngs"],
        })

    nz = gdf.loc[gdf["pred_crash_per_km"]>0,"pred_crash_per_km"]
    p5  = float(nz.quantile(0.05)) if len(nz) else 0.001
    p95 = float(nz.quantile(0.95)) if len(nz) else 1.0

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<title>{S.AREA_NAME} – Dynamic Risk | DriveFactor</title>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1.0">
{S.LEAFLET_HEAD}
<style>
{S.BASE_CSS}
#ctrl-panel{{ min-width:230px; max-width:250px }}
.mult-display{{font-size:18px;font-weight:800;color:#f59e0b;text-align:center;margin:4px 0 8px}}
.mult-label{{font-size:10px;color:#64748b;text-align:center;margin-bottom:8px}}
</style>
</head>
<body>
{S.TOPBAR_HTML.format(product="Dynamic Road Risk — Crashes / km / hr", area=S.AREA_NAME)}
<div style="position:relative;flex:1 1 0;display:flex;flex-direction:column">
<div id="map"></div>

<div id="ctrl-panel">
  <h3>Conditions</h3>

  <div class="ctrl-row"><label>Month</label><span class="ctrl-val" id="mo-lbl">March</span></div>
  <input type="range" id="sl-month" min="1" max="12" value="3" step="1" oninput="recalc()">

  <div class="ctrl-row"><label>Hour</label><span class="ctrl-val" id="hr-lbl">08:00</span></div>
  <input type="range" id="sl-hour" min="0" max="23" value="8" step="1" oninput="recalc()">

  <div class="ctrl-row"><label>Day type</label></div>
  <select id="sl-daytype" onchange="recalc()">
    <option value="Weekday">Weekday</option>
    <option value="Weekend">Weekend</option>
  </select>
  <div style="margin-bottom:8px"></div>

  <div class="ctrl-row"><label>Rainfall</label><span class="ctrl-val" id="rain-lbl">Dry</span></div>
  <input type="range" id="sl-rain" min="0" max="6" value="0" step="1" oninput="recalc()">

  <div class="ctrl-row"><label>Wind gusts</label><span class="ctrl-val" id="wind-lbl">Calm</span></div>
  <input type="range" id="sl-wind" min="0" max="5" value="0" step="1" oninput="recalc()">

  <div style="margin-top:10px;padding-top:10px;border-top:1px solid #334155">
    <div class="mult-display" id="mult-lbl">—</div>
    <div class="mult-label">Avg risk / km / hr</div>
  </div>
</div>

<div id="legend">
  <h4>Crash risk / km / hr</h4>
  <div class="leg-grad" style="background:{S.LEGEND_GRADIENT}"></div>
  <div class="leg-ticks"><span>Low</span><span>Med</span><span>High</span></div>
  <div class="leg-note">crash_rate/veh × hourly flow × conditions<br>multiplier (V4 GAM: weather × time × road type).</div>
</div>
</div>

<script>
{S.COLOR_RAMP_JS}
const FEATS = {json.dumps(feats)};
const V4 = {json.dumps(v4_lut)};
const P_BINS = {json.dumps(PRECIP_BINS)};
const G_BINS = {json.dumps(GUST_BINS)};
const P_LBLS = {json.dumps(PRECIP_LBLS)};
const G_LBLS = {json.dumps(GUST_LBLS)};
const MO_LBLS = ['','Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'];
const BASE_P5 = {p5}, BASE_P95 = {p95};

// Typical UK diurnal flow profile — fraction of AADF in each hour
// Weekday: AM peak 08:00, PM peak 17:00; Weekend: flatter midday peak
const HOURLY_PROFILE = {{
  Weekday: [0.012,0.007,0.005,0.005,0.009,0.022,0.052,0.082,0.071,0.058,0.056,0.057,
            0.058,0.057,0.060,0.068,0.085,0.082,0.062,0.051,0.042,0.034,0.025,0.017],
  Weekend: [0.016,0.010,0.007,0.006,0.007,0.013,0.023,0.036,0.052,0.063,0.071,0.074,
            0.074,0.072,0.069,0.066,0.063,0.058,0.051,0.045,0.038,0.032,0.025,0.019]
}};

let lines = [];

const map = L.map('map').setView([{S.CENTER_LAT},{S.CENTER_LON}],13);
L.tileLayer('https://{{s}}.basemaps.cartocdn.com/light_all/{{z}}/{{x}}/{{y}}{{r}}.png',{{
  attribution:'&copy; OpenStreetMap &copy; CartoDB',maxZoom:19
}}).addTo(map);

FEATS.forEach(f=>{{
  if(!f.latlngs||!f.latlngs.length) return;
  const w = f.highway==='motorway'||f.highway==='trunk'?5:f.highway==='primary'||f.highway==='secondary'?4:3;
  const line = L.polyline(f.latlngs,{{color:'#d1d5db',weight:w,opacity:0.88}}).addTo(map);
  line._feat = f; line._w = w;
  lines.push(line);
}});

function v4Mult(rt,dt,mo,hr,pi,gi){{
  const pk=String(Math.round(P_BINS[pi]*100)), gk=String(G_BINS[gi]);
  const m=V4[rt]?.[dt]?.[String(mo)]?.[String(hr)]?.[pk]?.[gk];
  return (m!==undefined)?m:1.0;
}}

function recalc(){{
  const mo = parseInt(document.getElementById('sl-month').value);
  const hr = parseInt(document.getElementById('sl-hour').value);
  const dt = document.getElementById('sl-daytype').value;
  const pi = parseInt(document.getElementById('sl-rain').value);
  const gi = parseInt(document.getElementById('sl-wind').value);

  document.getElementById('mo-lbl').textContent = MO_LBLS[mo];
  document.getElementById('hr-lbl').textContent = String(hr).padStart(2,'0')+':00';
  document.getElementById('rain-lbl').textContent = P_LBLS[pi].split(' ')[0];
  document.getElementById('wind-lbl').textContent = G_LBLS[gi].split(' ')[0];

  // Hourly flow fraction for this day type and hour
  const profKey = (dt === 'Weekend') ? 'Weekend' : 'Weekday';
  const flowFrac = HOURLY_PROFILE[profKey][hr];

  let sumRisk=0, n=0;
  lines.forEach(l=>{{
    const f=l._feat;
    const mult = v4Mult(f.road_type,dt,mo,hr,pi,gi);
    // hourly_flow = aadf * flowFrac  (vehicles in this hour)
    // crash_risk_per_km_per_hr = crash_rate_per_veh × hourly_flow × conditions_mult
    // Fallback: if no aadf/crash_rate_per_veh use base_cpk × mult × flowFrac × 24
    let risk_per_km_hr;
    if (f.crash_rate_per_veh > 0 && f.aadf > 0) {{
      risk_per_km_hr = f.crash_rate_per_veh * f.aadf * flowFrac * mult;
    }} else {{
      risk_per_km_hr = f.base_cpk * mult * flowFrac * 24 / 8760;
    }}
    const t = Math.max(0, Math.min(1, (risk_per_km_hr - BASE_P5) / (BASE_P95 - BASE_P5)));
    l.setStyle({{color:_rampColor(t), weight:l._w, opacity:0.88}});
    const label = f.road_name||f.ref_type||f.highway;
    l.bindTooltip(`
      <div class="tt-title">${{label}}</div>
      <div class="tt-sub">${{f.highway}} · ${{f.len_km.toFixed(2)}} km · ${{f.speed_limit||'?'}} mph</div>
      <hr class="tt-div">
      <div class="tt-row"><span>AADF (vehicles/day)</span><span class="tt-val">${{Math.round(f.aadf).toLocaleString()}}</span></div>
      <div class="tt-row"><span>Hourly flow</span><span class="tt-val">${{Math.round(f.aadf * flowFrac)}} veh/hr</span></div>
      <div class="tt-row"><span>Conditions multiplier</span><span class="tt-val">${{mult.toFixed(3)}}×</span></div>
      <div class="tt-row"><span>Risk / km / hr</span><span class="tt-val">${{risk_per_km_hr.toExponential(3)}}</span></div>
    `,{{className:'df-tooltip',sticky:true}});
    l.on('mouseover',function(){{this.setStyle({{weight:this._w+2}})}});
    l.on('mouseout',function(){{l.setStyle({{color:_rampColor(t),weight:l._w,opacity:0.88}})}});
    sumRisk+=risk_per_km_hr; n++;
  }});
  const avg = n>0?(sumRisk/n):0;
  const el=document.getElementById('mult-lbl');
  el.textContent=avg.toExponential(2);
  el.style.color=avg>BASE_P95*0.5?'#ef4444':avg>BASE_P95*0.2?'#f59e0b':'#22c55e';
}}

recalc();
</script>
</body>
</html>"""
    return html


def main():
    log.info("Loading segments …")
    gdf = S.load_segments()
    log.info("Loading V4 multiplier LUT …")
    v4_lut = S.load_v4_lut()
    log.info("Rendering dynamic risk map …")
    html = build(gdf, v4_lut)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(html, encoding="utf-8")
    log.info("Saved → %s", OUT)


if __name__ == "__main__":
    main()

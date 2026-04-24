"""render_sunglare.py – Sun-glare risk map with hour-of-day slider.

Product 3 of 6 – Stockton-on-Tees DriveFactor data-product showcase.
Shows per-segment glare risk for a single demo day (spring equinox, DOY 80 ≈ 21 March).
Hour slider updates segment colours in real time.
Output: outputs/data_products/sunglare.html
"""
from __future__ import annotations
import json, logging, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import shared as S

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

OUT    = S.OUT_DIR / "sunglare.html"
DEMO_DOY = 78   # ~19 March — near spring equinox, peak glare risk (nearest sampled DOY)


def build(gdf, glare_lut: dict):
    feats = []
    for _, row in gdf.iterrows():
        if not row["latlngs"]:
            continue
        sid = str(int(row["osm_id"]))
        # Build per-hour glare dict for the demo DOY only
        doy_str = str(DEMO_DOY)
        glare_by_hour = glare_lut.get(sid, {}).get(doy_str, {})
        feats.append({
            "osm_id":    int(row["osm_id"]),
            "road_name": str(row.get("road_name","") or ""),
            "ref_type":  str(row.get("ref_type","") or ""),
            "highway":   str(row.get("highway","") or ""),
            "glare":     {int(k): round(v,2) for k,v in glare_by_hour.items()},
            "pred_cpk":  round(float(row["pred_crash_per_km"]),5),
            "latlngs":   row["latlngs"],
        })

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<title>{S.AREA_NAME} – Sun Glare Risk | DriveFactor</title>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1.0">
{S.LEAFLET_HEAD}
<style>
{S.BASE_CSS}
#ctrl-panel{{ min-width:230px }}
.hour-display{{font-size:22px;font-weight:800;color:#2563eb;text-align:center;margin:4px 0 10px}}
</style>
</head>
<body>
{S.TOPBAR_HTML.format(product="Sun Glare Risk", area=S.AREA_NAME)}
<div style="position:relative;flex:1 1 0;display:flex;flex-direction:column">
<div id="map"></div>

<div id="ctrl-panel">
  <h3>Time of Day</h3>
  <div class="hour-display" id="hour-lbl">07:00</div>
  <input type="range" id="hour-sl" min="0" max="23" value="7" step="1" oninput="onHour()">
  <div style="display:flex;justify-content:space-between;font-size:10px;color:#64748b;margin-top:-3px">
    <span>00:00</span><span>12:00</span><span>23:00</span>
  </div>
  <div style="margin-top:10px;font-size:10px;color:#6b7280;line-height:1.5">
    Demo date: 19 March (spring equinox).<br>
    Glare risk peaks at low sun angles during<br>morning &amp; evening commute hours.
  </div>
</div>

<div id="legend">
  <h4>Glare risk score</h4>
  <div class="leg-grad" style="background:{S.LEGEND_GRADIENT_GLARE}"></div>
  <div class="leg-ticks"><span>0</span><span>50</span><span>100</span></div>
  <div class="leg-note">Computed from solar azimuth/elevation<br>vs road bearing. 0–100 scale.</div>
</div>
</div>

<script>
{S.COLOR_RAMP_JS}
const FEATS = {json.dumps(feats)};
let curH = 7;
let lines = [];

const map = L.map('map').setView([{S.CENTER_LAT},{S.CENTER_LON}],13);
L.tileLayer('https://{{s}}.basemaps.cartocdn.com/light_all/{{z}}/{{x}}/{{y}}{{r}}.png',{{
  attribution:'&copy; OpenStreetMap &copy; CartoDB',maxZoom:19
}}).addTo(map);

FEATS.forEach(f=>{{
  if(!f.latlngs||!f.latlngs.length) return;
  const w = f.highway==='motorway'||f.highway==='trunk'?5:f.highway==='primary'||f.highway==='secondary'?4:3;
  const line = L.polyline(f.latlngs,{{color:'#d1d5db',weight:w,opacity:0.9}}).addTo(map);
  line._feat = f;
  line._w = w;
  lines.push(line);
  line.on('mouseover',function(){{this.setStyle({{weight:this._w+2}})}});
  line.on('mouseout',function(){{updateLine(this)}});
}});

function glareAt(f,h){{
  const g = f.glare[h];
  return (g===undefined||g===null) ? 0 : g;
}}

function updateLine(line){{
  const f=line._feat;
  const g=glareAt(f,curH);
  const hasData = (g !== null && g !== undefined && g > 0);
  const col = hasData ? _glareRamp(g/100) : '#d1d5db';
  line.setStyle({{color:col,weight:line._w,opacity:hasData?0.9:0.5}});
  const label=f.road_name||f.ref_type||f.highway;
  const gLabel=g>0?g.toFixed(1):'No data';
  const risk=g>=70?'High':g>=35?'Moderate':g>0?'Low':'—';
  line.bindTooltip(`
    <div class="tt-title">${{label}}</div>
    <div class="tt-sub">${{f.highway}}</div>
    <hr class="tt-div">
    <div class="tt-row"><span>Glare score</span><span class="tt-val">${{gLabel}}</span></div>
    <div class="tt-row"><span>Risk level</span><span class="tt-val">${{risk}}</span></div>
    <div class="tt-row"><span>Pred. crash density</span><span class="tt-val">${{f.pred_cpk.toFixed(3)}}/km/yr</span></div>
  `,{{className:'df-tooltip',sticky:true}});
}}

function onHour(){{
  curH = parseInt(document.getElementById('hour-sl').value);
  const h = String(curH).padStart(2,'0');
  document.getElementById('hour-lbl').textContent = h+':00';
  lines.forEach(l=>updateLine(l));
}}

onHour();
</script>
</body>
</html>"""
    return html


def main():
    log.info("Loading segments …")
    gdf = S.load_segments()
    osm_ids = gdf["osm_id"].tolist()
    log.info("Loading glare data …")
    glare_lut = S.load_glare(osm_ids)
    log.info("Rendering sun-glare map …")
    html = build(gdf, glare_lut)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(html, encoding="utf-8")
    log.info("Saved → %s", OUT)


if __name__ == "__main__":
    main()

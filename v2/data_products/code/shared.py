"""shared.py – Common data loading utilities for Stockton-on-Tees data products.

All scripts in this directory import from here to ensure consistent data,
styling constants, and colour helpers.
"""
from __future__ import annotations
import json, logging, warnings
from pathlib import Path
import numpy as np
import pandas as pd
import psycopg2
import geopandas as gpd
from sqlalchemy import create_engine

warnings.filterwarnings("ignore")
log = logging.getLogger(__name__)

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT     = Path(__file__).resolve().parents[3]          # repo root
OUT_DIR  = Path(__file__).resolve().parents[1]           # outputs/data_products/
ML_ROOT  = Path("/Users/henry/Desktop/drivefactor-ml")
PREDS_CSV = ROOT / "scripts/precciitons/fold_2_predictions.csv"
V4_GRID   = ML_ROOT / "models/DYNAMIC_V4/outputs/v4_multiplier_grid_annual_normalised.csv"
V4_MEAN   = ML_ROOT / "models/DYNAMIC_V4/outputs/v4_multiplier_annual_normalised.csv"
HIER_OR   = ML_ROOT / "models/TRAFFIC_FLOW/casecrossover/outputs/hierarchical_or_curves.json"

DB_URL    = "postgresql://henry@localhost:5432/osm"
DB_CFG    = dict(dbname="osm", user="henry", host="localhost", port="5432")

# ── Stockton-on-Tees bounding box ─────────────────────────────────────────────
# Stockton-on-Tees bounding box — used only for map initial view / fallback
AREA_NAME  = "Stockton-on-Tees"
BBOX       = (-1.52, 54.47, -1.15, 54.64)   # approximate, true boundary used in queries
CENTER_LAT = 54.563
CENTER_LON = -1.329

# Road types to include
KEEP_HIGHWAY = (
    "motorway","motorway_link","trunk","trunk_link",
    "primary","primary_link","secondary","secondary_link",
    "tertiary","tertiary_link","unclassified","residential",
)

# ── Tile URL ──────────────────────────────────────────────────────────────────
TILE_URL = "https://{{s}}.basemaps.cartocdn.com/light_all/{{z}}/{{x}}/{{y}}{{r}}.png"
TILE_ATTR = "&copy; OpenStreetMap &copy; CartoDB"

# ── DriveFactor brand palette ─────────────────────────────────────────────────
BRAND = {
    "bg":        "#f0f2f5",
    "surface":   "#ffffff",
    "border":    "#e5e7eb",
    "text":      "#1f2937",
    "muted":     "#6b7280",
    "accent":    "#2563eb",
    "blue":      "#2563eb",
    "green":     "#16a34a",
    "red":       "#dc2626",
    "orange":    "#f97316",
}

# ── Colour ramps matching tees_valley_aimsun_risk_v2 ─────────────────────────
# JS snippet — included in every HTML template.
# _rampColor(t)   → green→yellow→red  (crash density, glare, alerting)
# _riskRamp(t)    → blue→green→yellow→red  (absolute risk / VKT)
# _blueRamp(t)    → light-blue→dark-blue  (legacy blue-only scale)
# _divRamp(t)     → blue→white→red diverging (OR, dynamic multiplier)
COLOR_RAMP_JS = """
function hexToRgb(h){
  const r=parseInt(h.slice(1,3),16),g=parseInt(h.slice(3,5),16),b=parseInt(h.slice(5,7),16);
  return [r,g,b];
}
function interpStops(stops,t){
  t=Math.max(0,Math.min(1,t));
  const n=stops.length-1;
  const idx=Math.min(Math.floor(t*n),n-1);
  const f=t*n-idx;
  const [r1,g1,b1]=hexToRgb(stops[idx]);
  const [r2,g2,b2]=hexToRgb(stops[idx+1]);
  return `rgb(${Math.round(r1+f*(r2-r1))},${Math.round(g1+f*(g2-g1))},${Math.round(b1+f*(b2-b1))})`;
}
// green → yellow → red  (crash density / risk maps)
function _rampColor(t){ return interpStops(['#1a9641','#a6d96a','#ffffbf','#fdae61','#d7191c'],t); }
// blue → green → yellow → red  (absolute risk / VKT — full safety ramp)
function _riskRamp(t){ return interpStops(['#2166ac','#1a9641','#ffffbf','#d7191c'],t); }
// grey → yellow → orange → red  (sun glare — no green)
function _glareRamp(t){ return interpStops(['#d1d5db','#e5e7eb','#fef08a','#f97316','#dc2626'],t); }
// light-blue → dark-blue  (absolute risk / VKT)
function _blueRamp(t){ return interpStops(['#deebf7','#9ecae1','#4292c6','#2171b5','#084594'],t); }
// diverging blue→white→red  (OR curves / dynamic multiplier)
function _divRamp(t){ return interpStops(['#2166ac','#74add1','#f7f7f7','#f46d43','#a50026'],t); }
// alias for OR (same diverging ramp)
function _orToT(v){const lo=0.7,hi=3.0;if(v<=lo)return 0;if(v>=hi)return 1;return Math.log(v/lo)/Math.log(hi/lo);}
function colOR(v){ return _divRamp(_orToT(v)); }
"""

# CSS gradient strings for legend bars
LEGEND_GRADIENT        = "linear-gradient(to right,#1a9641,#a6d96a,#ffffbf,#fdae61,#d7191c)"
LEGEND_GRADIENT_GLARE  = "linear-gradient(to right,#d1d5db,#e5e7eb,#fef08a,#f97316,#dc2626)"
LEGEND_GRADIENT_RISK   = "linear-gradient(to right,#2166ac,#1a9641,#ffffbf,#d7191c)"
LEGEND_GRADIENT_BLUE   = "linear-gradient(to right,#deebf7,#9ecae1,#4292c6,#2171b5,#084594)"
LEGEND_GRADIENT_DIV    = "linear-gradient(to right,#2166ac,#74add1,#f7f7f7,#f46d43,#a50026)"

# Shared CSS head fragment — light theme matching the original
BASE_CSS = """
*{box-sizing:border-box;margin:0;padding:0}
body{background:#f0f2f5;color:#1f2937;font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif;height:100vh;overflow:hidden;display:flex;flex-direction:column}
#topbar{flex:0 0 auto;background:#ffffff;border-bottom:1px solid #e5e7eb;padding:0 16px;display:flex;align-items:center;gap:12px;height:48px}
#topbar .logo{font-size:15px;font-weight:800;color:#1f2937;letter-spacing:-.3px}
#topbar .logo span{color:#2563eb}
#topbar .badge{background:#2563eb;color:#fff;font-size:10px;font-weight:700;padding:2px 7px;border-radius:3px;text-transform:uppercase}
#topbar .product-name{font-size:13px;font-weight:600;color:#4b5563}
#topbar .sep{flex:1}
#topbar .area{font-size:11px;color:#9ca3af}
#map{flex:1 1 0;z-index:0}
.df-tooltip{background:#ffffff !important;border:1px solid #d1d5db !important;border-radius:8px !important;color:#1f2937 !important;font-size:12px;padding:10px 12px;min-width:200px;max-width:280px;box-shadow:0 4px 16px rgba(0,0,0,.10) !important}
.tt-title{font-size:13px;font-weight:700;color:#111827;margin-bottom:4px}
.tt-sub{font-size:10px;color:#6b7280;margin-bottom:7px}
.tt-row{display:flex;justify-content:space-between;margin-bottom:3px;color:#374151;font-size:11px}
.tt-val{font-weight:700;color:#111827}
hr.tt-div{border:none;border-top:1px solid #e5e7eb;margin:5px 0}
#legend{position:absolute;bottom:18px;right:14px;z-index:999;background:rgba(255,255,255,.95);border:1px solid #d1d5db;border-radius:8px;padding:10px 13px;min-width:160px;font-size:11px;color:#374151;box-shadow:0 4px 16px rgba(0,0,0,.08)}
#legend h4{font-size:10px;font-weight:700;color:#6b7280;text-transform:uppercase;letter-spacing:.6px;margin-bottom:7px}
.leg-grad{width:100%;height:10px;border-radius:3px;margin-bottom:4px}
.leg-ticks{display:flex;justify-content:space-between;font-size:9px;color:#9ca3af;margin-bottom:6px}
.leg-note{font-size:10px;color:#9ca3af;line-height:1.4;margin-top:4px}
#ctrl-panel{position:absolute;top:14px;left:14px;z-index:999;background:rgba(255,255,255,.95);border:1px solid #d1d5db;border-radius:8px;padding:11px 13px;min-width:210px;font-size:12px;color:#374151;box-shadow:0 4px 16px rgba(0,0,0,.08)}
#ctrl-panel h3{font-size:10px;font-weight:700;color:#6b7280;text-transform:uppercase;letter-spacing:.6px;margin-bottom:8px}
.ctrl-row{display:flex;align-items:center;justify-content:space-between;margin-bottom:6px}
.ctrl-row label{font-size:12px;color:#6b7280;flex:1}
.ctrl-val{font-size:12px;font-weight:700;color:#111827;min-width:46px;text-align:right}
input[type=range]{width:100%;accent-color:#2563eb;margin:2px 0 5px}
select{width:100%;padding:5px 7px;border:1px solid #d1d5db;border-radius:5px;font-size:12px;background:#f9fafb;color:#111827}
"""

LEAFLET_HEAD = """
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"/>
<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
"""

TOPBAR_HTML = """
<div id="topbar">
  <span class="logo">Drive<span>Factor</span></span>
  <span class="badge">DEMO</span>
  <span class="product-name">{product}</span>
  <span class="sep"></span>
  <span class="area">{area} &nbsp;&middot;&nbsp; April 2026</span>
</div>
"""


# ── Data loaders ─────────────────────────────────────────────────────────────

def get_engine():
    return create_engine(DB_URL)

def get_conn():
    return psycopg2.connect(**DB_CFG)


# OSM polygon ID for Stockton-on-Tees local authority (admin_level=6)
STOCKTON_OSM_ID = -150994

def load_segments() -> gpd.GeoDataFrame:
    """Load Stockton-on-Tees OSM road segments using the true LA boundary polygon."""
    engine = get_engine()
    hw = ",".join(f"'{h}'" for h in KEEP_HIGHWAY)

    # Subquery: true Stockton-on-Tees boundary from osm_2025_polygon
    boundary_sql = (
        f"(SELECT ST_Transform(way,4326) FROM osm_2025_polygon "
        f"WHERE osm_id={STOCKTON_OSM_ID} LIMIT 1)"
    )

    sql = f"""
        SELECT DISTINCT ON (r.osm_id)
               r.osm_id,
               COALESCE(r.highway,'')  AS highway,
               COALESCE(r.name,'')     AS road_name,
               COALESCE(r.ref,'')      AS ref_type,
               COALESCE(r.maxspeed,'') AS maxspeed,
               COALESCE(r.lanes,'')    AS lanes,
               COALESCE(r.oneway,'')   AS oneway,
               ST_Length(ST_Transform(r.way,27700))/1000.0 AS osm_len_km,
               ST_Transform(r.way,4326) AS geometry
        FROM osm_2025_roads r
        WHERE r.highway IN ({hw})
          AND ST_Intersects(ST_Transform(r.way,4326), {boundary_sql})
        ORDER BY r.osm_id
    """
    gdf = gpd.read_postgis(sql, engine, geom_col="geometry")

    # Road class enrichment
    rc = pd.read_sql(
        f"SELECT osm_id,area_type,lane_class,highway AS hw_rc,maxspeed AS ms_rc "
        f"FROM osm_segment_road_class WHERE osm_id IN (SELECT osm_id FROM osm_2025_roads r "
        f"WHERE r.highway IN ({hw}) AND ST_Intersects(ST_Transform(r.way,4326), {boundary_sql}))",
        engine,
    )
    gdf = gdf.merge(rc, on="osm_id", how="left")

    # Predictions
    preds = pd.read_csv(
        PREDS_CSV,
        usecols=["osm_id","predicted_crashes","actual_crashes",
                 "road_type","area_type","speed_limit",
                 "predicted_percentile","actual_percentile"],
    )
    # predicted_crashes / actual_crashes are 4-year totals
    preds["annual_crashes"] = preds["predicted_crashes"] / 4.0
    gdf = gdf.merge(preds, on="osm_id", how="left")
    gdf["annual_crashes"] = gdf["annual_crashes"].fillna(0)

    # Calibration factor so sum(pred) == sum(obs)
    _sum_pred = gdf.drop_duplicates("osm_id")["annual_crashes"].sum()
    _sum_obs  = gdf.drop_duplicates("osm_id")["actual_crashes"].fillna(0).sum()
    calib = (_sum_obs / _sum_pred) if _sum_pred > 0 else 1.0
    gdf["annual_crashes_calib"] = gdf["annual_crashes"] * calib
    gdf["pred_crash_per_km"] = (
        gdf["annual_crashes_calib"] / gdf["osm_len_km"].replace(0, np.nan)
    ).fillna(0)

    # AADF from traffic v4 for abs risk
    conn = get_conn()
    ids_str = ",".join(str(int(x)) for x in gdf["osm_id"].tolist())
    tv4 = pd.read_sql(
        f"SELECT osm_id, aadf FROM osm_segment_traffic_v4 WHERE osm_id IN ({ids_str})",
        conn,
    )
    conn.close()
    tv4 = tv4.groupby("osm_id")["aadf"].mean().reset_index()
    gdf = gdf.merge(tv4, on="osm_id", how="left")
    gdf["aadf"] = gdf["aadf"].fillna(0)
    # annual_vkt = aadf * length_km * 365 (vehicles * km)
    gdf["annual_vkt"] = gdf["aadf"] * gdf["osm_len_km"] * 365
    gdf["abs_risk_per_mvkt"] = (
        gdf["annual_crashes_calib"] / (gdf["annual_vkt"] / 1e6).replace(0, np.nan)
    ).fillna(0)

    # Observed crash counts
    conn2 = get_conn()
    cr = pd.read_sql(
        "SELECT osm_id, COUNT(*) AS crash_count, "
        "SUM(CASE WHEN collision_severity='1' THEN 1 ELSE 0 END) AS fatal, "
        "SUM(CASE WHEN collision_severity='2' THEN 1 ELSE 0 END) AS serious, "
        "SUM(CASE WHEN collision_severity='3' THEN 1 ELSE 0 END) AS slight "
        f"FROM crashes WHERE osm_id IN ({ids_str}) GROUP BY osm_id",
        conn2,
    )
    conn2.close()
    gdf = gdf.merge(cr, on="osm_id", how="left")
    for col in ("crash_count","fatal","serious","slight"):
        gdf[col] = gdf[col].fillna(0).astype(int)
    gdf["crash_per_km"] = (
        gdf["crash_count"] / gdf["osm_len_km"].replace(0, np.nan)
    ).fillna(0)

    # Parse speed limit
    def _spd(ms):
        try: return int(str(ms).split()[0])
        except: return 0
    gdf["speed_limit"] = gdf["maxspeed"].apply(_spd)

    # Centroid lat/lon
    gdf["lat"] = gdf.geometry.centroid.y
    gdf["lon"] = gdf.geometry.centroid.x

    # latlngs list for JS polyline
    def _coords(geom):
        try:
            from shapely.geometry import MultiLineString, LineString
            if isinstance(geom, MultiLineString):
                coords = list(geom.geoms[0].coords)
            else:
                coords = list(geom.coords)
            return [[round(y,5), round(x,5)] for x,y in coords]
        except:
            return []
    gdf["latlngs"] = gdf.geometry.apply(_coords)

    log.info("Segments loaded: %d  (pred: %d  glare-ready)", len(gdf), int((gdf["pred_crash_per_km"]>0).sum()))
    return gdf


def load_glare(osm_ids: list) -> dict:
    """Return {str(osm_id): {str(doy): {str(hour): glare_risk}}} for full year."""
    conn = get_conn()
    ids_str = ",".join(str(int(x)) for x in osm_ids)
    df = pd.read_sql(
        f"SELECT osm_id, day_of_year, hour, glare_risk "
        f"FROM segment_glare_samples WHERE osm_id IN ({ids_str}) "
        f"ORDER BY osm_id, day_of_year, hour",
        conn,
    )
    conn.close()
    lut: dict = {}
    for _, row in df.iterrows():
        sid = str(int(row["osm_id"]))
        doy = str(int(row["day_of_year"]))
        hr  = str(int(row["hour"]))
        lut.setdefault(sid, {}).setdefault(doy, {})[hr] = round(float(row["glare_risk"]), 2)
    log.info("Glare LUT: %d segments, %d rows", len(lut), len(df))
    return lut


def load_v4_lut() -> dict:
    """Load V4 multiplier grid → nested dict road_type→day_type→month→hour→precip→gust."""
    if V4_GRID.exists():
        df = pd.read_csv(V4_GRID)
        log.info("V4 grid: %d rows, road types: %s", len(df), sorted(df["road_type"].unique().tolist()))
    else:
        log.warning("V4 grid missing, falling back to climate-mean")
        df = pd.read_csv(V4_MEAN)
        df["precip_bin_mid"] = 0.0
        df["gust_bin_mid"]   = 12.0
    lut: dict = {}
    for _, row in df.iterrows():
        rt = str(row["road_type"])
        dt = str(row["day_type"])
        mo = str(int(row["month"]))
        hr = str(int(row["hour"]))
        pk = str(int(round(float(row.get("precip_bin_mid", 0)) * 100)))
        gk = str(int(round(float(row.get("gust_bin_mid", 12)))))
        mv = round(float(row["multiplier_annual_norm"]), 5)
        lut.setdefault(rt,{}).setdefault(dt,{}).setdefault(mo,{}).setdefault(hr,{}).setdefault(pk,{})[gk] = mv
    return lut


def load_hier_or() -> dict:
    with open(HIER_OR) as fh:
        raw = json.load(fh)
    xs = raw["spd_sweep"]
    out: dict = {}
    for rg, periods in raw["cells"].items():
        out[rg] = {}
        for period, cell in periods.items():
            ors = cell["or_hierarchical"]
            out[rg][period] = {"spd_mids": xs, "or": ors, "n_crashes": cell["n_crashes"]}
    return out


def percentile_norm(series: pd.Series, lo_p=5, hi_p=95) -> pd.Series:
    """Map values to [0,1] using robust percentile bounds."""
    nz = series[series > 0]
    lo = float(nz.quantile(lo_p/100)) if len(nz) else 0
    hi = float(nz.quantile(hi_p/100)) if len(nz) else 1
    if hi == lo: hi = lo + 1e-9
    return ((series - lo) / (hi - lo)).clip(0, 1)

#!/usr/bin/env python3
"""render_intervention.py – Road Safety Intervention Planner (30→20 mph).

Precomputes per-segment predicted crash impact for Stockton-on-Tees using an
RF log-RR model and renders a self-contained Leaflet map.

Outputs: outputs/data_products/intervention.html

Usage:
  python render_intervention.py
"""
from __future__ import annotations

import json
import logging
import math
import os
import sys
import warnings
warnings.filterwarnings('ignore')
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import shared as S

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

OUT = S.OUT_DIR / "intervention.html"

# ── Model path ────────────────────────────────────────────────────────────────
_RF_MODEL = S.ML_ROOT / "models/INTERVENTION/outputs/hte/rf_log_rr_model.joblib"

# ── Crash year range ──────────────────────────────────────────────────────────
# All available years (Stats19 earliest reliable match to OSM 2025 segments)
_CRASH_YEAR_MIN  = 2014
_CRASH_YEAR_MAX  = 2024
_COVID_YEARS     = {2020, 2021}
_CLEAN_YEARS     = (_CRASH_YEAR_MAX - _CRASH_YEAR_MIN + 1) - len(_COVID_YEARS)  # = 9
_CRASH_LABEL     = f"{_CRASH_YEAR_MIN}–{_CRASH_YEAR_MAX} (excl. 2020–21)"

HIGHWAY_RANK = {
    "motorway": 1, "trunk": 2, "primary": 3, "secondary": 4,
    "tertiary": 5, "unclassified": 6, "residential": 7,
    "living_street": 8, "road": 9,
}

_ML_FEAT_COLS = [
    'highway_rank', 'lanes', 'log_length_m', 'sinuosity',
    'log_aadf', 'k1_neighbor_aadf_mean', 'population_density',
    'junction_count_1km', 'signal_count_mean', 'crossing_count_mean',
    'upstream_junc_deg', 'k1_neighbor_speed_mean',
    'pre_crash_rate_ckm',
]


# ── Model loader ───────────────────────────────────────────────────────────────

def load_rf_model():
    import joblib
    if not _RF_MODEL.exists():
        logger.warning("RF model not found at %s", _RF_MODEL)
        return None, None, None
    bundle = joblib.load(_RF_MODEL)
    logger.info("Loaded RF model (%d features)", len(bundle["feat_cols"]))
    return bundle["model"], bundle["scaler"], bundle["feat_cols"]


# ── DB fetchers ───────────────────────────────────────────────────────────────

def _latlngs_from_geojson(geojson_str: str):
    """Convert ST_AsGeoJSON LineString → [[lat,lon], …] for Leaflet."""
    try:
        g = json.loads(geojson_str)
        coords = g["coordinates"]
        if g["type"] == "MultiLineString":
            coords = coords[0]
        return [[round(y, 5), round(x, 5)] for x, y in coords]
    except Exception:
        return []


def fetch_stockton_segments(conn) -> pd.DataFrame:
    cur = conn.cursor()
    boundary_subq = (
        f"(SELECT way FROM osm_2025_polygon "
        f"WHERE osm_id={S.STOCKTON_OSM_ID} LIMIT 1)"
    )
    cur.execute(f"""
        SELECT
            r.osm_id,
            COALESCE(r.name, 'Unnamed road')                          AS name,
            r.highway,
            r.maxspeed,
            COALESCE(r.lanes, '1')                                     AS lanes_str,
            ST_AsGeoJSON(ST_Transform(r.way, 4326))                    AS geojson,
            ST_Length(ST_Transform(r.way, 27700))                      AS length_m,
            ST_Distance(
                ST_Transform(ST_StartPoint(r.way), 27700),
                ST_Transform(ST_EndPoint(r.way),   27700)
            ) / NULLIF(ST_Length(ST_Transform(r.way, 27700)), 0)      AS sinuosity,
            ST_X(ST_Transform(ST_Centroid(r.way), 4326))               AS lon,
            ST_Y(ST_Transform(ST_Centroid(r.way), 4326))               AS lat
        FROM osm_2025_roads r
        WHERE ST_Intersects(r.way, {boundary_subq})
        AND r.highway IN (
            'residential','tertiary','secondary','unclassified',
            'living_street','primary','trunk'
        )
    """)
    rows = cur.fetchall()
    cur.close()
    cols = ["osm_id","name","highway","maxspeed","lanes_str",
            "geojson","length_m","sinuosity","lon","lat"]
    df = pd.DataFrame(rows, columns=cols)
    df["latlngs"] = df["geojson"].apply(_latlngs_from_geojson)
    logger.info("Fetched %d segments", len(df))
    return df


def fetch_crash_counts(conn, osm_ids: list) -> pd.DataFrame:
    """All clean years: 2014-2024 excl. COVID (2020-21)."""
    cur = conn.cursor()
    cur.execute("""
        SELECT osm_id,
               COUNT(*) FILTER (
                   WHERE collision_year BETWEEN %s AND %s
                     AND collision_year NOT IN (2020, 2021)
               )  AS crashes_clean,
               COUNT(*) FILTER (
                   WHERE collision_year BETWEEN %s AND %s
               )  AS crashes_total
        FROM crashes
        WHERE osm_id = ANY(%s)
          AND collision_year BETWEEN %s AND %s
        GROUP BY osm_id
    """, (_CRASH_YEAR_MIN, _CRASH_YEAR_MAX,
          _CRASH_YEAR_MIN, _CRASH_YEAR_MAX,
          osm_ids,
          _CRASH_YEAR_MIN, _CRASH_YEAR_MAX))
    rows = cur.fetchall()
    cur.close()
    return pd.DataFrame(rows, columns=["osm_id","crashes_clean","crashes_total"])


def fetch_aadf(conn, osm_ids: list) -> pd.DataFrame:
    cur = conn.cursor()
    cur.execute("""
        SELECT DISTINCT ON (osm_id) osm_id, aadf
        FROM osm_segment_traffic_v4
        WHERE osm_id = ANY(%s)
        ORDER BY osm_id, year DESC NULLS LAST
    """, (osm_ids,))
    rows = cur.fetchall()
    cur.close()
    return pd.DataFrame(rows, columns=["osm_id","aadf"])


def fetch_crossing_density(conn, osm_ids: list) -> pd.DataFrame:
    cur = conn.cursor()
    cur.execute("""
        SELECT osm_id,
               COALESCE(upstream_junction_degree, 0)                          AS crossing_density,
               COALESCE(upstream_junction_degree, 0)                          AS upstream_junction_degree,
               COALESCE((upstream_junction_degree+downstream_junction_degree)/2.0, 0) AS segment_junction_complexity
        FROM osm_segment_junctions WHERE osm_id = ANY(%s)
    """, (osm_ids,))
    rows = cur.fetchall()
    cur.close()
    cols = ["osm_id","crossing_density","upstream_junction_degree","segment_junction_complexity"]
    return pd.DataFrame(rows, columns=cols) if rows else pd.DataFrame(columns=cols)


def fetch_population_density(conn, osm_ids: list) -> pd.DataFrame:
    cur = conn.cursor()
    cur.execute("SELECT osm_id, population_density FROM osm_segment_population_density WHERE osm_id = ANY(%s)", (osm_ids,))
    rows = cur.fetchall()
    cur.close()
    return pd.DataFrame(rows, columns=["osm_id","population_density"]) if rows else pd.DataFrame(columns=["osm_id","population_density"])


def fetch_network_features(conn, osm_ids: list) -> pd.DataFrame:
    cur = conn.cursor()
    cur.execute("""
        SELECT osm_id,
               k1_neighbor_aadf_mean,
               k1_neighbor_speed_mean,
               k1_neighbor_signal_count_mean       AS signal_count_mean,
               k1_neighbor_crossing_count_mean     AS crossing_count_mean,
               k1_neighbor_upstream_junction_degree_mean AS upstream_junc_deg
        FROM osm_segment_network WHERE osm_id = ANY(%s)
    """, (osm_ids,))
    rows = cur.fetchall()
    cur.close()
    cols = ["osm_id","k1_neighbor_aadf_mean","k1_neighbor_speed_mean",
            "signal_count_mean","crossing_count_mean","upstream_junc_deg"]
    return pd.DataFrame(rows, columns=cols) if rows else pd.DataFrame(columns=cols)


def fetch_junction_density(conn, osm_ids: list) -> pd.DataFrame:
    return pd.DataFrame(columns=["osm_id","junction_count_1km"])


# ── Feature engineering ────────────────────────────────────────────────────────

def build_features(segs, crashes, aadf, crossings, population, network=None, junction=None):
    df = segs.copy()

    def _parse_lanes(s):
        try: return int(str(s).split(";")[0].strip())
        except: return 1

    df["lanes"]        = df["lanes_str"].apply(_parse_lanes).clip(1, 6)
    df["highway_rank"] = df["highway"].map(HIGHWAY_RANK).fillna(7).astype(int)
    df["sinuosity"]    = df["sinuosity"].fillna(1.0).clip(0.1, 1.0)
    df["length_m"]     = df["length_m"].fillna(50.0).clip(1)
    df["log_length_m"] = np.log1p(df["length_m"])
    df["length_km"]    = (df["length_m"] / 1000.0).clip(lower=0.01)

    df = df.merge(crashes, on="osm_id", how="left").drop_duplicates("osm_id")
    df["crashes_clean"] = df["crashes_clean"].fillna(0)
    df["crashes_total"] = df["crashes_total"].fillna(0)

    df["pre_crash_rate_ckm"] = df["crashes_clean"] / _CLEAN_YEARS / df["length_km"]
    df["pre_crash_rate"]     = df["crashes_clean"] / _CLEAN_YEARS

    df = df.merge(aadf.drop_duplicates("osm_id"), on="osm_id", how="left").drop_duplicates("osm_id")
    df["aadf"]     = df["aadf"].fillna(0)
    df["log_aadf"] = np.log1p(df["aadf"])

    df = df.merge(crossings.drop_duplicates("osm_id"), on="osm_id", how="left").drop_duplicates("osm_id")
    for c in ["crossing_density","upstream_junction_degree","segment_junction_complexity"]:
        df[c] = df.get(c, pd.Series(0.0, index=df.index)).fillna(0.0)

    df = df.merge(population.drop_duplicates("osm_id"), on="osm_id", how="left").drop_duplicates("osm_id")
    df["population_density"] = df["population_density"].fillna(df["population_density"].median())

    if network is not None and len(network) > 0:
        df = df.merge(network.drop_duplicates("osm_id"), on="osm_id", how="left")
    for c in ["k1_neighbor_aadf_mean","k1_neighbor_speed_mean","signal_count_mean","crossing_count_mean","upstream_junc_deg"]:
        if c not in df.columns: df[c] = 0.0
        df[c] = df[c].fillna(0.0)

    if junction is not None and len(junction) > 0:
        df = df.merge(junction.drop_duplicates("osm_id"), on="osm_id", how="left")
    if "junction_count_1km" not in df.columns: df["junction_count_1km"] = 0.0
    df["junction_count_1km"] = df["junction_count_1km"].fillna(0.0)

    return df


# ── RF prediction ─────────────────────────────────────────────────────────────

_RF_FALLBACK_PCT = 19.7   # global mean used if RF model unavailable

def predict_impact(segments_df, rf_model=None, scaler=None, rf_feat_cols=None):
    df = segments_df.copy()

    col_remap = {"pre_crash_rate_per_km": "pre_crash_rate_ckm"}
    rf_pcts = np.full(len(df), np.nan)

    if rf_model is not None and scaler is not None and rf_feat_cols:
        X_rows = []
        for _, row in df.iterrows():
            X_rows.append([float(row.get(col_remap.get(fc, fc), 0) or 0) for fc in rf_feat_cols])
        X = scaler.transform(np.array(X_rows))
        log_rr_pred = np.clip(rf_model.predict(X), np.log(0.01), 0.0)

        _SPATIAL_CV_R2  = 0.31
        _DID_ATT_LOG_RR = np.log(1.0 - 0.204)
        _MAX_LOG_RR     = np.log(1.0 - 0.40)
        log_rr_pred = np.clip(
            _SPATIAL_CV_R2 * log_rr_pred + (1.0 - _SPATIAL_CV_R2) * _DID_ATT_LOG_RR,
            _MAX_LOG_RR, 0.0
        )
        rf_pcts = (1.0 - np.exp(log_rr_pred)) * 100.0
        logger.info("RF applied to %d segments", len(df))
    else:
        logger.warning("RF model unavailable — using global fallback %.1f%%", _RF_FALLBACK_PCT)
        rf_pcts[:] = _RF_FALLBACK_PCT

    predictions = []
    for i, (_, row) in enumerate(df.iterrows()):
        pre_annual = float(row.get("pre_crash_rate", 0))
        pre_ckm    = float(row.get("pre_crash_rate_ckm", 0))
        pct        = round(float(rf_pcts[i]), 1)

        pred_change = round(pre_annual * (-pct / 100.0), 4)
        post_annual = round(max(0.0, pre_annual + pred_change), 3)

        predictions.append({
            "osm_id":            int(row["osm_id"]),
            "predicted_change":  pred_change,
            "pre_annual":        round(pre_annual, 3),
            "pre_annual_ckm":    round(pre_ckm, 3),
            "post_annual":       post_annual,
            "pct_change":        round(-pct, 1),
            "risk_reduction_pct": pct,
        })

    result = pd.DataFrame(predictions)
    logger.info("Predicted impact for %d segments (mean=%.1f%%)",
                len(result), result["risk_reduction_pct"].mean())
    return result


# ── Corridor grouping ─────────────────────────────────────────────────────────

def _snap(coord, precision=4):
    """Snap a (lat, lon) pair to a grid to merge near-identical endpoints."""
    return (round(coord[0], precision), round(coord[1], precision))


def build_corridors(merged: pd.DataFrame) -> pd.DataFrame:
    """Group segments into corridors using name + topology.

    Rules:
      1. Named segments are grouped only when they share BOTH the same
         normalised name AND a snapped endpoint — so 'Norton Road' in the
         north and a disconnected 'Norton Road' elsewhere stay separate.
      2. Unnamed / junction segments are kept as their own single-segment
         corridor — they never drag named roads together.
    """
    df = merged.copy().reset_index(drop=True)

    def _norm_name(n):
        n = str(n or "").strip().lower()
        return n if n and n != "unnamed road" else ""

    df["_nname"] = df["name"].apply(_norm_name)

    # Union-Find over row indices
    parent = list(range(len(df)))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        a, b = find(a), find(b)
        if a != b:
            parent[b] = a

    # Build endpoint → list of (row_index, norm_name) — named only
    node_to_named: dict = defaultdict(list)
    for i, row in df.iterrows():
        nm = row["_nname"]
        if not nm:
            continue
        ll = row.get("latlngs") or []
        if len(ll) < 2:
            continue
        node_to_named[_snap(ll[0])].append((i, nm))
        node_to_named[_snap(ll[-1])].append((i, nm))

    # Only union segments that share an endpoint AND the same name
    for segs_at_node in node_to_named.values():
        by_name: dict = defaultdict(list)
        for idx, nm in segs_at_node:
            by_name[nm].append(idx)
        for same_name_segs in by_name.values():
            for i in range(1, len(same_name_segs)):
                union(same_name_segs[0], same_name_segs[i])

    df["_cid"] = [find(i) for i in range(len(df))]

    records = []
    for cid, grp in df.groupby("_cid"):
        total_len = grp["length_m"].sum() or 1.0
        w = grp["length_m"] / total_len

        is_30 = bool(grp["is_30mph"].any())

        rr_vals = grp.loc[grp["is_30mph"], "risk_reduction_pct"]
        rr_w    = grp.loc[grp["is_30mph"], "length_m"]
        red_pct = float((rr_vals * (rr_w / rr_w.sum())).sum()) if (is_30 and len(rr_vals)) else float("nan")

        names = [n for n in grp["name"].tolist() if n and n not in ("Unnamed road", "")]
        name  = max(set(names), key=names.count) if names else "Unnamed road"

        records.append({
            "corridor_id":        int(cid),
            "name":               name,
            "highway":            grp["highway"].mode()[0],
            "length_m":           round(total_len, 1),
            "lanes":              int(grp["lanes"].max()),
            "aadf":               int(grp["aadf"].mean()),
            "crashes_total":      int(grp["crashes_total"].sum()),
            "pre_annual":         round(float(grp["pre_annual"].sum()), 3),
            "post_annual":        round(float(grp["post_annual"].sum()), 3),
            "pct_change":         round(float((grp["pct_change"].fillna(0) * w).sum()), 1),
            "risk_reduction_pct": round(red_pct, 1) if not (isinstance(red_pct, float) and math.isnan(red_pct)) else float("nan"),
            "is_30mph":           is_30,
            "segment_count":      len(grp),
            "all_latlngs":        [r["latlngs"] for _, r in grp.iterrows() if r.get("latlngs")],
        })

    corridors = pd.DataFrame(records)

    # Rank-percentile colour within 30 mph corridors
    mask = corridors["risk_reduction_pct"].notna() & corridors["is_30mph"]
    corridors["color_t"] = float("nan")
    corridors.loc[mask, "color_t"] = corridors.loc[mask, "risk_reduction_pct"].rank(pct=True)

    logger.info("Grouped %d segments → %d corridors (%d with 30 mph prediction)",
                len(df), len(corridors), int(mask.sum()))
    return corridors


# ── HTML builder ───────────────────────────────────────────────────────────────

def build_html(segs: pd.DataFrame, preds: pd.DataFrame, summary: dict) -> str:
    merged = segs.merge(preds, on="osm_id", how="left")

    # is_30mph flag needed by build_corridors
    merged["is_30mph"] = (
        merged["maxspeed"].str.startswith("30", na=True) |
        merged["highway"].isin(["residential","living_street"])
    )

    corridors = build_corridors(merged)

    # Legend percentile ticks from corridor-level data
    mask = corridors["risk_reduction_pct"].notna() & corridors["is_30mph"]
    pct_series = corridors.loc[mask, "risk_reduction_pct"]
    _pct_lo  = round(float(pct_series.quantile(0.05)), 1) if len(pct_series) else 0.0
    _pct_mid = round(float(pct_series.quantile(0.50)), 1) if len(pct_series) else 20.0
    _pct_hi  = round(float(pct_series.quantile(0.95)), 1) if len(pct_series) else 40.0

    def _f(v, default=0.0):
        try: return float(v) if v is not None and not (isinstance(v, float) and math.isnan(v)) else default
        except: return default

    feats = []
    for _, row in corridors.iterrows():
        all_ll = row.get("all_latlngs") or []
        if not all_ll:
            continue
        feats.append({
            "name":               str(row.get("name") or ""),
            "highway":            str(row.get("highway") or ""),
            "length_m":           round(_f(row.get("length_m")), 0),
            "lanes":              int(row.get("lanes") or 1),
            "aadf":               int(row.get("aadf") or 0),
            "crashes_total":      int(row.get("crashes_total") or 0),
            "pre_annual":         round(_f(row.get("pre_annual")), 3),
            "post_annual":        round(_f(row.get("post_annual")), 3),
            "pct_change":         round(_f(row.get("pct_change")), 1),
            "risk_reduction_pct": round(_f(row.get("risk_reduction_pct")), 1),
            "is_30mph":           bool(row.get("is_30mph")),
            "color_t":            round(_f(row.get("color_t")), 4),
            "segment_count":      int(row.get("segment_count") or 1),
            "all_latlngs":        all_ll,
        })

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<title>{S.AREA_NAME} – Intervention Planner | DriveFactor</title>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1.0">
{S.LEAFLET_HEAD}
<style>
{S.BASE_CSS}
body{{overflow:hidden}}
#layout{{flex:1 1 0;display:flex;overflow:hidden}}
#sidebar{{flex:0 0 280px;background:#f9fafb;border-right:1px solid #e5e7eb;overflow-y:auto;display:flex;flex-direction:column;font-size:12px}}
.sb-sec{{padding:10px 13px;border-bottom:1px solid #e5e7eb}}
.sb-sec h3{{font-size:10px;font-weight:700;color:#6b7280;text-transform:uppercase;letter-spacing:.6px;margin-bottom:7px}}
.sb-empty{{color:#9ca3af;font-size:12px;padding:18px 13px;line-height:1.5}}
.tt-row{{display:flex;justify-content:space-between;margin-bottom:4px;color:#374151;font-size:11px}}
.tt-val{{font-weight:700;color:#111827}}
#map-wrap{{flex:1 1 0;position:relative}}
#map{{width:100%;height:100%}}
#wi-legend{{position:absolute;bottom:14px;right:12px;z-index:999;background:rgba(255,255,255,.95);border:1px solid #d1d5db;border-radius:8px;padding:9px 12px;min-width:170px;font-size:11px;color:#374151;box-shadow:0 4px 16px rgba(0,0,0,.08)}}
#wi-legend h4{{font-size:10px;font-weight:700;color:#6b7280;text-transform:uppercase;letter-spacing:.5px;margin-bottom:6px}}
.leg-grad{{width:100%;height:10px;border-radius:3px;margin-bottom:4px}}
.leg-ticks{{display:flex;justify-content:space-between;font-size:9px;color:#9ca3af;margin-bottom:4px}}
.leg-note{{font-size:10px;color:#9ca3af;line-height:1.4;margin-top:4px}}
/* Before/after bar chart */
.bar-chart{{margin:10px 0 4px}}
.bar-row{{display:flex;align-items:center;gap:6px;margin-bottom:5px;font-size:11px}}
.bar-label{{width:52px;color:#6b7280;flex-shrink:0;text-align:right}}
.bar-track{{flex:1;background:#e5e7eb;border-radius:3px;height:14px;position:relative;overflow:hidden}}
.bar-fill{{height:100%;border-radius:3px;transition:width .4s}}
.bar-val{{width:44px;font-size:10px;font-weight:700;color:#111827;flex-shrink:0}}
</style>
</head>
<body>
{S.TOPBAR_HTML.format(product="Road Safety Intervention Planner — 30→20 mph Impact", area=S.AREA_NAME)}
<div id="layout">

<div id="sidebar">
  <div class="sb-sec">
    <h3>Network Summary</h3>
    <div class="tt-row"><span>30 mph segments</span><span class="tt-val">{summary["segments_30mph"]:,}</span></div>
    <div class="tt-row"><span>Avg crash reduction</span><span class="tt-val">{abs(summary["avg_pct_change"]):.1f}%</span></div>
    <div style="margin:10px 0 2px">
      <div class="bar-row">
        <div class="bar-label" style="width:62px">Before</div>
        <div class="bar-track"><div class="bar-fill" style="width:100%;background:#f97316"></div></div>
        <div class="bar-val">{summary["total_pre_annual"]:.1f} /yr</div>
      </div>
      <div class="bar-row">
        <div class="bar-label" style="width:62px">After</div>
        <div class="bar-track"><div class="bar-fill" style="width:{round(summary["total_post_annual"] / summary["total_pre_annual"] * 100) if summary["total_pre_annual"] else 100:.0f}%;background:#16a34a"></div></div>
        <div class="bar-val">{summary["total_post_annual"]:.1f} /yr</div>
      </div>
    </div>
    <div class="tt-row" style="margin-top:6px"><span>Crashes avoidable / yr</span><span class="tt-val" style="color:#16a34a">−{summary["total_crashes_avoidable_per_yr"]:.1f}</span></div>
  </div>
  <div class="sb-sec" style="flex:1">
    <h3>Segment Detail</h3>
    <div id="sb-content" class="sb-empty">Click a coloured road to see the predicted impact of a 30→20 mph reduction.</div>
  </div>
</div>

<div id="map-wrap">
  <div id="map"></div>
  <div id="wi-legend">
    <h4>Predicted crash reduction</h4>
    <div class="leg-grad" style="background:linear-gradient(to right,#dc2626,#f97316,#fef08a,#4ade80,#16a34a)"></div>
    <div class="leg-ticks"><span>{_pct_lo:.0f}%</span><span></span><span>{_pct_mid:.0f}%</span><span></span><span>{_pct_hi:.0f}%</span></div>
    <div class="leg-note">Colour = rank within network.<br>Grey = not 30 mph.</div>
  </div>
</div>
</div>

<script>
{S.COLOR_RAMP_JS}

// Red→yellow→green, matching the glare map palette (crisp, saturated stops)
function reductionColor(t){{
  return interpStops(['#dc2626','#f97316','#fef08a','#4ade80','#16a34a'], t);
}}

const FEATS = {json.dumps(feats)};

const map = L.map('map', {{zoomControl:true}}).setView([{S.CENTER_LAT},{S.CENTER_LON}],13);
L.tileLayer('https://{{s}}.basemaps.cartocdn.com/light_all/{{z}}/{{x}}/{{y}}{{r}}.png',{{
  attribution:'&copy; OpenStreetMap &copy; CartoDB', maxZoom:19
}}).addTo(map);

// Load Stockton boundary
fetch('https://nominatim.openstreetmap.org/search?q=Stockton-on-Tees&format=json&polygon_geojson=1&limit=1')
  .catch(()=>{{}});  // boundary overlay optional — silently skip if blocked

FEATS.forEach(f => {{
  if (!f.all_latlngs || !f.all_latlngs.length) return;
  const color   = f.is_30mph ? reductionColor(f.color_t) : '#d1d5db';
  const opacity = f.is_30mph ? 0.92 : 0.28;
  const w = f.highway==='motorway'||f.highway==='trunk'?6
          : f.highway==='primary'||f.highway==='secondary'?5:4;

  // Render every constituent polyline segment of this corridor
  f.all_latlngs.forEach(ll => {{
    if (!ll || !ll.length) return;
    const line = L.polyline(ll, {{color, weight:w, opacity}}).addTo(map);
    line.on('mouseover', () => line.setStyle({{weight:w+2, opacity:1}}));
    line.on('mouseout',  () => line.setStyle({{weight:w, opacity:f.is_30mph?0.92:0.28}}));
    line.on('click', () => showSegment(f));
  }});
}});

function barHtml(label, value, maxVal, color){{
  const w = maxVal > 0 ? Math.round(value / maxVal * 100) : 0;
  return `<div class="bar-row">
    <div class="bar-label">${{label}}</div>
    <div class="bar-track"><div class="bar-fill" style="width:${{w}}%;background:${{color}}"></div></div>
    <div class="bar-val">${{value.toFixed(3)}}</div>
  </div>`;
}}

function showSegment(f) {{
  if (!f.is_30mph) {{
    document.getElementById('sb-content').innerHTML =
      '<div style="color:#9ca3af;font-size:12px;line-height:1.5">This road is not currently 30 mph — no intervention modelled.</div>';
    return;
  }}
  const pct    = f.risk_reduction_pct;
  const label  = f.name || f.highway;
  const maxVal = Math.max(f.pre_annual, 0.001);
  const lenKm  = (f.length_m / 1000).toFixed(2);

  document.getElementById('sb-content').innerHTML = `
    <div style="font-size:13px;font-weight:700;color:#111827;margin-bottom:1px">${{label}}</div>
    <div style="font-size:10px;color:#6b7280;margin-bottom:10px">${{f.highway}} &nbsp;·&nbsp; ${{lenKm}} km &nbsp;·&nbsp; ${{f.segment_count}} section${{f.segment_count>1?'s':''}}</div>

    <div style="font-size:28px;font-weight:800;color:#16a34a;text-align:center;line-height:1">${{pct.toFixed(1)}}%</div>
    <div style="font-size:10px;color:#6b7280;text-align:center;margin-bottom:10px">predicted crash reduction</div>

    <div class="bar-chart">
      ${{barHtml('Before', f.pre_annual,  maxVal, '#f97316')}}
      ${{barHtml('After',  f.post_annual, maxVal, '#16a34a')}}
    </div>
    <div style="font-size:10px;color:#9ca3af;margin-bottom:8px">Predicted crashes / yr across corridor</div>

    <hr style="border:none;border-top:1px solid #e5e7eb;margin:6px 0">
    <div class="tt-row"><span>Recorded crashes</span><span class="tt-val">${{f.crashes_total}}</span></div>
    <div class="tt-row"><span>Daily traffic (AADF)</span><span class="tt-val">${{f.aadf > 0 ? f.aadf.toLocaleString() : 'n/a'}}</span></div>
  `;
}}
</script>
</body>
</html>"""


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    conn = S.get_conn()

    logger.info("Fetching Stockton segments …")
    segs    = fetch_stockton_segments(conn)
    osm_ids = segs["osm_id"].tolist()

    logger.info("Fetching crash counts (%s) …", _CRASH_LABEL)
    crashes    = fetch_crash_counts(conn, osm_ids)
    aadf       = fetch_aadf(conn, osm_ids)
    crossings  = fetch_crossing_density(conn, osm_ids)
    population = fetch_population_density(conn, osm_ids)
    network    = fetch_network_features(conn, osm_ids)
    junction   = fetch_junction_density(conn, osm_ids)
    conn.close()

    rf_model, scaler, rf_feats = load_rf_model()
    logger.info("Building features …")
    segs_feat = build_features(segs, crashes, aadf, crossings, population, network, junction)

    segs_30 = segs_feat[
        segs_feat["maxspeed"].str.startswith("30", na=True) |
        segs_feat["highway"].isin(["residential","living_street"])
    ].copy()

    logger.info("Predicting impact for %d 30 mph segments …", len(segs_30))
    preds = predict_impact(segs_30,
                           rf_model=rf_model, scaler=scaler, rf_feat_cols=rf_feats)

    pred_segs = preds[preds["predicted_change"].notna()]
    summary = {
        "total_segments":              len(segs_feat),
        "segments_30mph":              len(segs_30),
        "avg_pct_change":              float(pred_segs["pct_change"].mean()) if len(pred_segs) else 0.0,
        "total_crashes_avoidable_per_yr": float(pred_segs["predicted_change"].clip(upper=0).abs().sum()),
        "total_pre_annual":            float(pred_segs["pre_annual"].sum()) if len(pred_segs) else 0.0,
        "total_post_annual":           float(pred_segs["post_annual"].sum()) if len(pred_segs) else 0.0,
        "crash_year_range":            _CRASH_LABEL,
    }

    logger.info("Rendering HTML …")
    html = build_html(segs_feat, preds, summary)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(html, encoding="utf-8")
    logger.info("Saved → %s  (%.1f MB)", OUT, OUT.stat().st_size / 1e6)

    logger.info("── Summary ─────────────────────────────────────────")
    logger.info("  30 mph segments:    %d", summary["segments_30mph"])
    logger.info("  Avg change:         %.1f%%", summary["avg_pct_change"])
    logger.info("  Crashes avoidable:  %.1f /yr", summary["total_crashes_avoidable_per_yr"])


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""
CESMLC V9.8.5 three-mode, window-mismatch, memory-efficient detector.

This is a clean branch from the useful V9.5/V9.6 extraction design.  It is not
a cumulative V11 rule stack.

Modes
-----
* ``signature_dominant`` (default) preserves the V9.8.4 decision behavior:
  deterministic IF evidence plus the existing signature decision tree.
* ``if_dominant`` admits an ECGI when a changed/model-scored feature reaches
  ``--if-dominant-percentile``. KMeans then describes admitted ECGIs, while
  the full-history signature mapper assigns one of the same four labels.
  Ordinary one-time RARE fallbacks are rejected unless the existing held-out
  same-age one-change conformal model confirms an independent extreme.
* ``compare`` runs both projections, produces a one-row-per-ECGI union with
  detector and cluster agreement, and writes two optional pie charts.

KMeans is diagnostic in every mode: it never admits, removes, or maps an ECGI.

Detection contract
------------------
1. By default, an ECGI becomes a candidate when DB_MISMATCH='YES' on any
   analysis-window date.  The original latest-day-only gate remains available.
2. Detection uses source/current history only; every *_DB column is ignored.
3. PHYSICALCELLID is removed before history engineering, modeling, clustering,
   scoring, and reporting.
4. SITE_TYPE is attached only after detection as output context.  COW, MACRO,
   UNKNOWN, and missing SITE_TYPE therefore use exactly the same pooled models
   and can never change a model score, cluster, suppression decision, or rank.
5. Candidate and sampled non-mismatch baseline ECGIs receive the same temporal
   feature engineering.  Normal change fields are never fabricated as zero.
6. The trailing quiet window (10 days by default) is recency context only.
   It never erases a qualifying pattern elsewhere in the full analysis window.
7. Exactly four final clusters are possible:
      HIGH_FREQUENCY_CHANGES  configured distinct-change-date threshold
      REAL_BACK_AND_FORTH     a real A->B->A/reverse transition
      MULTI_FEATURE_CHANGES   configured feature-count threshold across dates,
                              not a same-day-dominated coordinated step
      RARE_CHANGES            recurrent unusual behavior; a current/repeated
                              source conflict; or an ultra-tail one-step path
                              versus held-out same-age normal one-step history
8. Defaults, fallback values, location-shift magnitude, invalid-range rules,
   persistence aliases, and peer-rollout suppressors do not create or suppress
   an incident.
9. In Mode 1, scores only rank rule-qualified incidents. Mode 2 uses
   ``--if-dominant-percentile`` for raw model admission, then rejects only an
   ordinary one-time RARE fallback that lacks independently calibrated
   one-change evidence. Recurrent, BF, HF, and multi-feature paths remain.
10. Each feature Isolation Forest is a deterministic multi-seed ensemble with
    seed-specific held-out-normal calibration. Optional fine-tuning happens
    strictly inside the fit pool; candidates and the outer calibration holdout
    are never inspected during tuning.
11. Stable candidate-feature histories that have neither an inter-day change
    nor a same-day conflict stay in DuckDB and never enter pandas. This changes
    memory use and audit volume, not anomaly eligibility.

Suggested Windows command
-------------------------
python cesmlc_v985_three_mode_anomaly_detector.py `
  --input "C:\\Temp\\enmt_cesmlc_lte_audit_p_akr3aesme01_31d_withDB_v3.csv" `
  --cell-viewer-data "C:\\Temp\\cell_viewer_data.csv" `
  --strict-site-type-context `
  --candidate-gate window_any_mismatch `
  --candidate-gate-days 31 `
  --candidate-mismatch-min-days 1 `
  --days 30 `
  --stable-days 10 `
  --history-anchor-days 7 `
  --latlon-decimals 5 `
  --high-frequency-min-change-days 2 `
  --multi-feature-min-features 2 `
  --multi-feature-min-dates 2 `
  --out-dir "C:\\Temp\\v985_outputs" `
  --detection-mode compare `
  --if-dominant-percentile 98 `
  --kmeans-clusters 4 `
  --kmeans-rule-compatibility `
  --tune-isolation-forest `
  --isolation-seeds "17,42,73" `
  --isolation-n-estimators 400 `
  --isolation-max-samples 2048 `
  --isolation-max-features 0.80 `
  --normal-baseline-max-ecgis 50000 `
  --isolation-fit-sample 50000 `
  --memory-limit 10GB `
  --threads 4 `
  --feature-batch-size 1 `
  --reuse-history-cache `
  --force-rebuild-history-cache `
  --history-cache-dir "C:\\Temp\\cesmlc_v984_window_gate_cache" `
  --cache-match-mode compatible `
  --cache-input-identity basename
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import re
import shutil
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

SCRIPT_VERSION = "9.8.5-three-mode-window-mismatch-memory-efficient"
CACHE_SCHEMA = "v984-window-gate-active-candidate-feature-loading-modal-anchor-no-pci-v1"
FINAL_CLUSTERS = [
    "HIGH_FREQUENCY_CHANGES",
    "REAL_BACK_AND_FORTH",
    "MULTI_FEATURE_CHANGES",
    "RARE_CHANGES",
]
CLUSTER_TIE_PRIORITY = {
    "REAL_BACK_AND_FORTH": 4,
    "HIGH_FREQUENCY_CHANGES": 3,
    "MULTI_FEATURE_CHANGES": 2,
    "RARE_CHANGES": 1,
}
DETECTION_MODES = [
    "signature_dominant",
    "if_dominant",
    "compare",
]
MODE1_LABEL = "IF + SIGNATURE"
MODE2_LABEL = "IF DOMINANCY"

DATE_CANDIDATES = ["ENMT_LOAD_DATE", "PULLDATE", "LOAD_DATE", "AUDIT_DATE", "DATE"]
META_COLS = ["REGION", "MARKETCLUSTER", "MARKET", "SUBMARKET", "SERVER_ID"]
ID_CONTEXT_COLS = ["MCC", "MNC", "CID", "USID", "CELLID", "CELL_ID", "NR_CELL_ID", "NRCELLID"]
SITE_TYPE_CANDIDATES = ["SITE_TYPE", "SITETYPE", "SITE_CLASS", "SITE_CLASSIFICATION", "SITE_CATEGORY"]
NULL_STRINGS = {"", "NULL", "(NULL)", "NAN", "NONE", "NA", "N/A", "<NA>", "-"}
FORBIDDEN_FEATURES = {"PHYSICALCELLID"}

# PHYSICALCELLID is deliberately absent.
DEFAULT_FEATURES = [
    "CELLNAME", "CELLTYPE", "ANTENNALAT", "ANTENNALON",
    "ANTENNAORIENTATION", "ANTENNAOPENING", "ANTENNAHORIZONTALRANGE",
    "ANTENNABASEALTITUDE", "GEOTYPE", "RADIUS1", "SRVNGAREAAVERAGEALT",
    "SRVNGAREAAVGALTUNCERT", "SUPPORTEDTECHNOLOGIES",
]

ID_LIKE_EXACT = {
    "ECGI", "USID", "SERVER_ID", "MCC", "MNC", "CID", "PCI", "PCID",
    "CELLID", "CELL_ID", "NR_CELL_ID", "NRCELLID",
}
NUMERIC_CONFIG_TERMS = [
    "LAT", "LON", "LONG", "ORIENTATION", "OPENING", "RANGE", "RADIUS",
    "ALTITUDE", "AVERAGEALT", "AVGALT", "UNCERT", "AZIMUTH", "BEARING",
    "SPACING",
]

MODEL_INPUT_COLUMNS = [
    "CHANGE_DAYS_RATE",
    "LOG_CHANGE_EVENTS",
    "LOG_DISTINCT_VALUES",
    "MODE_IMPURITY_FRACTION",
    "ABA_EVENT_RATE",
    "REVERSE_EVENT_RATE",
    "RECENT_CHANGE_DAYS_RATE",
    "LOG_RECENCY_WEIGHTED_ACTIVITY",
    "TRANSITION_REPEAT_RATE",
    "CHANGE_INTERVAL_IRREGULARITY",
]

RUN_START = time.time()


def log(message: str, indent: int = 0) -> None:
    elapsed = int(time.time() - RUN_START)
    hours, rem = divmod(elapsed, 3600)
    minutes, seconds = divmod(rem, 60)
    print(
        f"[{time.strftime('%H:%M:%S')} | +{hours:02d}:{minutes:02d}:{seconds:02d}] "
        f"{'  ' * indent}{message}",
        flush=True,
    )


def header(title: str) -> None:
    print("\n" + "=" * 100, flush=True)
    log(title)
    print("=" * 100, flush=True)


def clean(value: object) -> str:
    return str(value).strip().strip('"').upper()


def norm_key(value: object) -> str:
    if pd.isna(value):
        return ""
    text = str(value).strip().strip('"')
    if text.upper() in NULL_STRINGS:
        return ""
    return re.sub(r"\s+", "", re.sub(r"\.0$", "", text)).upper()


def norm_val(value: object) -> str:
    if pd.isna(value):
        return ""
    text = str(value).strip().strip('"')
    if text.upper() in NULL_STRINGS:
        return ""
    try:
        number = float(text)
        if math.isfinite(number):
            if abs(number - round(number)) < 1e-12:
                return str(int(round(number)))
            return f"{number:.10f}".rstrip("0").rstrip(".")
    except Exception:
        pass
    return re.sub(r"\s+", "", text).upper()


def safe_num(value: Any, default: float = 0.0, index: Optional[pd.Index] = None) -> pd.Series:
    if isinstance(value, pd.Series):
        series = value.copy()
    else:
        series = pd.Series(value, index=index) if index is not None else pd.Series(value)
    return pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(default)


def col_num(frame: pd.DataFrame, name: str, default: float = 0.0) -> pd.Series:
    if frame is not None and name in frame.columns:
        return safe_num(frame[name], default).reindex(frame.index).fillna(default)
    return pd.Series(default, index=frame.index, dtype=float)


def col_text(frame: pd.DataFrame, name: str, default: str = "") -> pd.Series:
    if frame is not None and name in frame.columns:
        return frame[name].fillna(default).astype(str).reindex(frame.index).fillna(default)
    return pd.Series(default, index=frame.index, dtype=object)


def scalar_float(value: Any, default: float = 0.0) -> float:
    """Convert one audit/report value without letting NaN leak into formatting."""
    try:
        number = float(value)
        return number if math.isfinite(number) else float(default)
    except Exception:
        return float(default)


def scalar_int(value: Any, default: int = 0) -> int:
    return int(round(scalar_float(value, float(default))))


def quote_ident(name: str) -> str:
    return '"' + str(name).replace('"', '""') + '"'


def sql_string(value: object) -> str:
    return "'" + str(value).replace("'", "''") + "'"


def sql_path(path: Path) -> str:
    return sql_string(str(path).replace("\\", "/"))


def base_expr(expr: str) -> str:
    normalized = f"UPPER(REGEXP_REPLACE(TRIM(CAST({expr} AS VARCHAR)), '\\.0$', ''))"
    nulls = ", ".join(sql_string(item) for item in sorted(NULL_STRINGS))
    return (
        f"CASE WHEN {expr} IS NULL THEN NULL "
        f"WHEN {normalized} IN ({nulls}) THEN NULL ELSE {normalized} END"
    )


def compact_expr(expr: str) -> str:
    base = base_expr(expr)
    return f"CASE WHEN {base} IS NULL THEN NULL ELSE REGEXP_REPLACE({base}, '\\s+', '', 'g') END"


def date_expr(original_column: str) -> str:
    q = quote_ident(original_column)
    text = f"TRIM(CAST({q} AS VARCHAR))"
    return "CAST(COALESCE(" + ",".join([
        f"TRY_CAST({q} AS TIMESTAMP)",
        f"TRY_STRPTIME({text}, '%Y-%m-%d %H:%M:%S')",
        f"TRY_STRPTIME({text}, '%Y-%m-%d')",
        f"TRY_STRPTIME({text}, '%m/%d/%Y %H:%M:%S')",
        f"TRY_STRPTIME({text}, '%m/%d/%Y')",
        f"TRY_STRPTIME({text}, '%d-%b-%y')",
        f"TRY_STRPTIME({text}, '%d-%b-%Y')",
    ]) + ") AS DATE)"


def is_id_like(column: str) -> bool:
    name = clean(column)
    return name in ID_LIKE_EXACT or name.endswith("_ID") or name.endswith("ID") or "CELLID" in name


def is_numeric_config(column: str) -> bool:
    name = clean(column)
    return (not is_id_like(name)) and any(term in name for term in NUMERIC_CONFIG_TERMS)


def feature_expr(original_column: str, feature: str, latlon_decimals: int) -> str:
    q = quote_ident(original_column)
    normalized = compact_expr(q)
    if is_numeric_config(feature):
        decimals = latlon_decimals if any(term in clean(feature) for term in ["LAT", "LON", "LONG"]) else 4
        return (
            f"CASE WHEN {normalized} IS NULL THEN NULL "
            f"WHEN TRY_CAST({normalized} AS DOUBLE) IS NOT NULL "
            f"THEN CAST(ROUND(TRY_CAST({normalized} AS DOUBLE), {int(decimals)}) AS VARCHAR) "
            f"ELSE {normalized} END"
        )
    return normalized


def ecgi_sql(colmap: Dict[str, str]) -> str:
    if "ECGI" in colmap:
        return compact_expr(quote_ident(colmap["ECGI"]))
    if all(name in colmap for name in ["MCC", "MNC", "CID"]):
        return (
            f"COALESCE({compact_expr(quote_ident(colmap['MCC']))}, '') || '-' || "
            f"COALESCE({compact_expr(quote_ident(colmap['MNC']))}, '') || '-' || "
            f"COALESCE({compact_expr(quote_ident(colmap['CID']))}, '')"
        )
    raise ValueError("No ECGI column and ECGI cannot be built from MCC/MNC/CID.")


def read_header(path: Path) -> Tuple[List[str], Dict[str, str]]:
    columns = list(pd.read_csv(path, nrows=0).columns)
    return [clean(column) for column in columns], {clean(column): column for column in columns}


def first_existing(columns: Sequence[str], candidates: Sequence[str]) -> Optional[str]:
    available = set(columns)
    return next((item for item in candidates if item in available), None)


def chunks(items: Sequence[str], size: int) -> Iterable[List[str]]:
    step = max(1, int(size))
    for start in range(0, len(items), step):
        yield list(items[start:start + step])


def detect_features(columns: List[str], include: Optional[str], exclude: Optional[str]) -> List[str]:
    available = set(columns)
    requested = (
        [clean(item) for item in include.split(",") if item.strip()]
        if include else [feature for feature in DEFAULT_FEATURES if feature in available]
    )
    excluded = {clean(item) for item in exclude.split(",") if item.strip()} if exclude else set()
    explicitly_forbidden = sorted(set(requested) & FORBIDDEN_FEATURES)
    if explicitly_forbidden:
        log(
            "Ignoring forbidden monitored feature(s): " + ", ".join(explicitly_forbidden)
            + ". PHYSICALCELLID cannot enter V9.8 history or detection.",
            1,
        )
    requested = [
        feature for feature in requested
        if feature not in FORBIDDEN_FEATURES and feature not in excluded and not feature.endswith("_DB")
    ]
    missing = [feature for feature in requested if feature not in available]
    if missing:
        raise ValueError(f"Selected monitored features are missing from the CESMLC CSV: {missing}")
    unique = list(dict.fromkeys(requested))
    if not unique:
        raise ValueError("No monitored source features remain after exclusions.")
    return unique


def import_duckdb():
    try:
        import duckdb  # type: ignore
        return duckdb
    except Exception as exc:
        raise RuntimeError("DuckDB is required: python -m pip install duckdb") from exc


def open_duckdb(db_path: Path, temp_dir: Path, threads: int, memory_limit: str):
    duckdb = import_duckdb()
    db_path.parent.mkdir(parents=True, exist_ok=True)
    temp_dir.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(db_path))
    try:
        con.execute("SET preserve_insertion_order=false")
    except Exception:
        pass
    con.execute(f"SET temp_directory={sql_string(str(temp_dir))}")
    con.execute(f"PRAGMA threads={max(1, int(threads))}")
    if memory_limit:
        con.execute(f"PRAGMA memory_limit={sql_string(memory_limit)}")
    return con


def full_file_fingerprint(path: Path, block_size: int = 8 * 1024 * 1024) -> str:
    """Hash every input byte so a compatible cache cannot miss middle-file edits."""
    started = time.perf_counter()
    log(f"Hashing complete CESMLC input for exact cache identity: {path}", 1)
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(block_size)
            if not block:
                break
            digest.update(block)
    fingerprint = digest.hexdigest()
    log(f"Complete input hash finished in {time.perf_counter() - started:,.1f}s.", 1)
    return fingerprint


def empirical_percentile(reference_scores: np.ndarray, scores: np.ndarray) -> np.ndarray:
    reference = np.sort(np.asarray(reference_scores, dtype=float))
    reference = reference[np.isfinite(reference)]
    values = np.asarray(scores, dtype=float)
    result = np.zeros(len(values), dtype=float)
    finite = np.isfinite(values)
    if reference.size == 0:
        return result
    left = np.searchsorted(reference, values[finite], side="left")
    right = np.searchsorted(reference, values[finite], side="right")
    result[finite] = 100.0 * (left + right) / (2.0 * float(reference.size))
    return result


def conformal_upper_tail_p(
    reference_scores: np.ndarray,
    scores: np.ndarray,
) -> np.ndarray:
    """Conservative upper-tail conformal p-values with a finite-sample +1."""
    reference = np.sort(np.asarray(reference_scores, dtype=float))
    reference = reference[np.isfinite(reference)]
    values = np.asarray(scores, dtype=float)
    result = np.ones(len(values), dtype=float)
    finite = np.isfinite(values)
    if reference.size == 0:
        return result
    less_than = np.searchsorted(reference, values[finite], side="left")
    greater_or_equal = reference.size - less_than
    result[finite] = (1.0 + greater_or_equal) / (reference.size + 1.0)
    return result


def one_change_age_bucket(values: pd.Series) -> pd.Series:
    """Predeclared recency strata used only for one-change normal calibration."""
    days = safe_num(values, 9999)
    categories = [
        "AGE_0_3D", "AGE_4_7D", "AGE_8_14D", "AGE_15_30D",
        "AGE_UNAVAILABLE",
    ]
    labels = np.select(
        [days.between(0, 3), days.between(4, 7),
         days.between(8, 14), days.between(15, 30)],
        ["AGE_0_3D", "AGE_4_7D", "AGE_8_14D", "AGE_15_30D"],
        default="AGE_UNAVAILABLE",
    )
    return pd.Series(
        pd.Categorical(labels, categories=categories), index=values.index
    )


@dataclass
class RunInfo:
    window_start: pd.Timestamp
    window_end: pd.Timestamp
    stable_start: pd.Timestamp
    candidate_count: int
    baseline_count: int
    candidate_hash: str
    baseline_hash: str
    latest_raw_rows: int
    latest_distinct_ecgis: int
    latest_snapshot_ratio: float
    site_type_match_ratio: float

    def as_dict(self) -> Dict[str, Any]:
        return {
            "script_version": SCRIPT_VERSION,
            "window_start_date": str(self.window_start.date()),
            "window_end_date": str(self.window_end.date()),
            "trailing_recency_context_start_date": str(self.stable_start.date()),
            "candidate_ecgi_count": self.candidate_count,
            "normal_baseline_ecgi_count": self.baseline_count,
            "candidate_hash": self.candidate_hash,
            "baseline_hash": self.baseline_hash,
            "latest_raw_rows": self.latest_raw_rows,
            "latest_distinct_ecgis": self.latest_distinct_ecgis,
            "latest_snapshot_ratio_to_prior_median": self.latest_snapshot_ratio,
            "candidate_site_type_match_ratio": self.site_type_match_ratio,
        }


def create_empty_site_type_map(con) -> None:
    """Create the typed empty context table used when the second source is absent."""
    con.execute("DROP TABLE IF EXISTS site_type_map")
    con.execute("""
        CREATE TABLE site_type_map(
            ECGI VARCHAR,
            SITE_TYPE VARCHAR,
            SITE_TYPE_MATCH_STATUS VARCHAR,
            IS_COW_SITE INTEGER,
            SITE_TYPE_SOURCE_ROW_COUNT BIGINT,
            SITE_TYPE_DISTINCT_COUNT BIGINT
        )
    """)


def build_site_type_map(
    con,
    cell_viewer_path: Optional[Path],
    strict: bool = False,
) -> Dict[str, int]:
    """Resolve context when possible; failure never removes an ECGI by default."""
    header("PHASE 0 - SITE_TYPE CONTEXT MAP")
    if cell_viewer_path is None:
        create_empty_site_type_map(con)
        log("SITE_TYPE source not supplied; every candidate will be retained as UNKNOWN.", 1)
        return {}
    try:
        columns, colmap = read_header(cell_viewer_path)
        site_column = first_existing(columns, SITE_TYPE_CANDIDATES)
        if not site_column:
            raise ValueError(
                f"cell_viewer_data has no SITE_TYPE column. Tried {SITE_TYPE_CANDIDATES}."
            )
        date_column = first_existing(columns, DATE_CANDIDATES)
        ecgi = ecgi_sql(colmap)
        site = compact_expr(quote_ident(colmap[site_column]))
        observed_date = date_expr(colmap[date_column]) if date_column else "NULL::DATE"
        source = sql_path(cell_viewer_path)
        con.execute("DROP TABLE IF EXISTS site_type_map")
        con.execute(f"""
        CREATE TABLE site_type_map AS
        WITH raw AS (
            SELECT {ecgi} ECGI, {site} SITE_TYPE, {observed_date} OBSERVED_DATE
            FROM read_csv_auto({source}, HEADER=TRUE, ALL_VARCHAR=TRUE, SAMPLE_SIZE=100000)
        ), valid AS (
            SELECT * FROM raw
            WHERE ECGI IS NOT NULL AND ECGI<>'' AND SITE_TYPE IS NOT NULL AND SITE_TYPE<>''
        ), type_stats AS (
            SELECT ECGI, SITE_TYPE, COUNT(*) TYPE_ROWS, MAX(OBSERVED_DATE) LAST_TYPE_DATE
            FROM valid GROUP BY ECGI, SITE_TYPE
        ), ecgi_stats AS (
            SELECT ECGI, SUM(TYPE_ROWS) SOURCE_ROWS, COUNT(*) DISTINCT_TYPES,
                   MAX(LAST_TYPE_DATE) MAX_TYPE_DATE
            FROM type_stats GROUP BY ECGI
        ), ranked AS (
            SELECT t.*, e.SOURCE_ROWS, e.DISTINCT_TYPES, e.MAX_TYPE_DATE,
                   ROW_NUMBER() OVER(
                       PARTITION BY t.ECGI
                       ORDER BY t.LAST_TYPE_DATE DESC NULLS LAST, t.TYPE_ROWS DESC, t.SITE_TYPE ASC
                   ) RN,
                   COUNT(*) FILTER (WHERE t.LAST_TYPE_DATE=e.MAX_TYPE_DATE)
                       OVER(PARTITION BY t.ECGI) LATEST_TYPE_COUNT
            FROM type_stats t JOIN ecgi_stats e USING(ECGI)
        ), picked AS (
            SELECT *, UPPER(REGEXP_REPLACE(SITE_TYPE, '[^A-Z0-9]', '', 'g')) SITE_TYPE_COMPACT
            FROM ranked WHERE RN=1
        )
        SELECT
            ECGI,
            CASE
                WHEN DISTINCT_TYPES>1 AND MAX_TYPE_DATE IS NULL THEN 'UNKNOWN'
                WHEN DISTINCT_TYPES>1 AND LATEST_TYPE_COUNT>1 THEN 'UNKNOWN'
                ELSE SITE_TYPE
            END SITE_TYPE,
            CASE
                WHEN DISTINCT_TYPES>1 AND (MAX_TYPE_DATE IS NULL OR LATEST_TYPE_COUNT>1)
                    THEN 'AMBIGUOUS'
                WHEN DISTINCT_TYPES=1 THEN 'DIRECT_UNIQUE'
                ELSE 'RESOLVED_LATEST'
            END SITE_TYPE_MATCH_STATUS,
            CASE
                WHEN NOT (DISTINCT_TYPES>1 AND (MAX_TYPE_DATE IS NULL OR LATEST_TYPE_COUNT>1))
                 AND (
                    SITE_TYPE_COMPACT='COW' OR SITE_TYPE_COMPACT LIKE 'COW%'
                    OR SITE_TYPE_COMPACT LIKE '%CELLONWHEELS%'
                    OR SITE_TYPE_COMPACT LIKE '%MOBILECELL%'
                 ) THEN 1 ELSE 0
            END IS_COW_SITE,
            SOURCE_ROWS SITE_TYPE_SOURCE_ROW_COUNT,
            DISTINCT_TYPES SITE_TYPE_DISTINCT_COUNT
        FROM picked
        """)
    except Exception as exc:
        if strict:
            raise
        create_empty_site_type_map(con)
        log(
            f"WARNING: SITE_TYPE context could not be read ({exc}); "
            "all candidates remain eligible and will be labeled UNKNOWN.",
            1,
        )
        return {}
    counts = {
        str(status): int(count)
        for status, count in con.execute(
            "SELECT SITE_TYPE_MATCH_STATUS, COUNT(*) FROM site_type_map GROUP BY SITE_TYPE_MATCH_STATUS"
        ).fetchall()
    }
    log("SITE_TYPE resolution: " + str(counts), 1)
    return counts


def build_candidate_site_context(
    con,
    run_info: RunInfo,
) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """Left-attach one context row to every gated candidate, including unmatched ECGIs."""
    context = con.execute("""
        SELECT
            c.ECGI,
            COALESCE(s.SITE_TYPE, 'UNKNOWN') SITE_TYPE,
            COALESCE(s.SITE_TYPE_MATCH_STATUS, 'UNMATCHED') SITE_TYPE_MATCH_STATUS,
            COALESCE(s.IS_COW_SITE, 0) IS_COW_SITE,
            COALESCE(s.SITE_TYPE_SOURCE_ROW_COUNT, 0) SITE_TYPE_SOURCE_ROW_COUNT,
            COALESCE(s.SITE_TYPE_DISTINCT_COUNT, 0) SITE_TYPE_DISTINCT_COUNT
        FROM candidate_ecgis c
        LEFT JOIN site_type_map s USING(ECGI)
        ORDER BY c.ECGI
    """).fetchdf()
    context.columns = [clean(column) for column in context.columns]
    if "ECGI" in context.columns:
        context["ECGI"] = context["ECGI"].map(norm_key)
    if len(context) != int(run_info.candidate_count):
        raise AssertionError(
            "SITE_TYPE context changed the candidate count; context must be a pure left attachment."
        )
    if not context.empty and not context["ECGI"].is_unique:
        raise AssertionError("SITE_TYPE context contains more than one row per candidate ECGI.")
    for column, default in {
        "SITE_TYPE": "UNKNOWN",
        "SITE_TYPE_MATCH_STATUS": "UNMATCHED",
        "IS_COW_SITE": 0,
        "SITE_TYPE_SOURCE_ROW_COUNT": 0,
        "SITE_TYPE_DISTINCT_COUNT": 0,
    }.items():
        if column not in context.columns:
            context[column] = default
    context["SITE_TYPE"] = col_text(context, "SITE_TYPE", "UNKNOWN")
    context["SITE_TYPE_MATCH_STATUS"] = col_text(
        context, "SITE_TYPE_MATCH_STATUS", "UNMATCHED"
    )
    for column in ["IS_COW_SITE", "SITE_TYPE_SOURCE_ROW_COUNT", "SITE_TYPE_DISTINCT_COUNT"]:
        context[column] = col_num(context, column, 0).astype(int)
    counts = {
        str(status): int(count)
        for status, count in context["SITE_TYPE_MATCH_STATUS"].value_counts(dropna=False).items()
    }
    if sum(counts.values()) != int(run_info.candidate_count):
        raise AssertionError("SITE_TYPE status counts do not cover every candidate ECGI.")
    matched = int(
        (~context["SITE_TYPE_MATCH_STATUS"].isin(["UNMATCHED", "AMBIGUOUS"])).sum()
    )
    run_info.site_type_match_ratio = matched / max(1, int(run_info.candidate_count))
    log(
        f"Candidate SITE_TYPE coverage={run_info.site_type_match_ratio:.1%}; "
        f"retained candidates={len(context):,}; dropped=0; status={counts}.",
        1,
    )
    return context, counts


def attach_site_type_context(
    frame: pd.DataFrame,
    candidate_context: pd.DataFrame,
) -> pd.DataFrame:
    """Attach output-only context without changing row count, order, or candidate set."""
    result = frame.copy(deep=False)
    context_columns = [
        "SITE_TYPE", "SITE_TYPE_MATCH_STATUS", "IS_COW_SITE",
        "SITE_TYPE_SOURCE_ROW_COUNT", "SITE_TYPE_DISTINCT_COUNT",
    ]
    for column in context_columns:
        if column in result.columns:
            del result[column]
    if "ECGI" not in result.columns:
        for column, default in {
            "SITE_TYPE": "UNKNOWN", "SITE_TYPE_MATCH_STATUS": "UNMATCHED",
            "IS_COW_SITE": 0, "SITE_TYPE_SOURCE_ROW_COUNT": 0,
            "SITE_TYPE_DISTINCT_COUNT": 0,
        }.items():
            result[column] = default
        return result
    row_count = len(result)
    original_index = result.index
    result["ECGI"] = result["ECGI"].map(norm_key)
    context = candidate_context[["ECGI", *context_columns]].copy()
    context["ECGI"] = context["ECGI"].map(norm_key)
    if context["ECGI"].duplicated().any():
        raise AssertionError("SITE_TYPE context must contain one row per candidate ECGI.")
    context = context.set_index("ECGI")
    for column in context_columns:
        result[column] = result["ECGI"].map(context[column])
    result["SITE_TYPE"] = col_text(result, "SITE_TYPE", "UNKNOWN")
    result["SITE_TYPE_MATCH_STATUS"] = col_text(
        result, "SITE_TYPE_MATCH_STATUS", "UNMATCHED"
    )
    for column in ["IS_COW_SITE", "SITE_TYPE_SOURCE_ROW_COUNT", "SITE_TYPE_DISTINCT_COUNT"]:
        result[column] = col_num(result, column, 0).astype(int)
    result["COW_REVIEW_NOTE"] = np.where(
        result["IS_COW_SITE"].eq(1),
        "COW SITE: review mobility as operational context; SITE_TYPE did not alter detection or score.",
        "",
    )
    if len(result) != row_count or not result.index.equals(original_index):
        raise AssertionError("SITE_TYPE attachment changed row count or row order.")
    return result


def export_site_type_audit(context: pd.DataFrame, path: Path) -> None:
    """Export one row for every candidate, including UNKNOWN/UNMATCHED rows."""
    path.parent.mkdir(parents=True, exist_ok=True)
    context.to_csv(path, index=False)


def candidate_site_context_sql(
    candidate_relation: str = "candidate_ecgis",
    include_gate_fields: bool = False,
) -> str:
    """Return the context-only left join used by both streaming export and final rows."""
    gate_fields = """
            c.GATE_MISMATCH_ROWS,
            c.GATE_MISMATCH_DAYS,
            c.GATE_FIRST_MISMATCH_DATE,
            c.GATE_LAST_MISMATCH_DATE,
            c.CANDIDATE_GATE_REASON,
    """ if include_gate_fields else ""
    return f"""
        SELECT
            c.ECGI,
            {gate_fields}
            COALESCE(s.SITE_TYPE, 'UNKNOWN') SITE_TYPE,
            COALESCE(s.SITE_TYPE_MATCH_STATUS, 'UNMATCHED') SITE_TYPE_MATCH_STATUS,
            COALESCE(s.IS_COW_SITE, 0) IS_COW_SITE,
            COALESCE(s.SITE_TYPE_SOURCE_ROW_COUNT, 0) SITE_TYPE_SOURCE_ROW_COUNT,
            COALESCE(s.SITE_TYPE_DISTINCT_COUNT, 0) SITE_TYPE_DISTINCT_COUNT
        FROM {candidate_relation} c
        LEFT JOIN site_type_map s USING(ECGI)
    """


def export_candidate_site_context_streaming(
    con,
    run_info: RunInfo,
    path: Path,
) -> Dict[str, int]:
    """Write every gated candidate directly from DuckDB without a giant pandas frame."""
    path.parent.mkdir(parents=True, exist_ok=True)
    query = candidate_site_context_sql(include_gate_fields=True)
    con.execute(
        f"COPY ({query} ORDER BY ECGI) TO {sql_path(path)} "
        "(FORMAT CSV, HEADER TRUE)"
    )
    distribution_path = path.with_name(
        "cesmlc_v984_candidate_gate_mismatch_day_distribution.csv"
    )
    con.execute(f"""
        COPY (
            SELECT
                GATE_MISMATCH_DAYS,
                COUNT(*) CANDIDATE_ECGIS,
                SUM(IS_COW_SITE) COW_CANDIDATE_ECGIS,
                COUNT(*) FILTER(
                    WHERE SITE_TYPE_MATCH_STATUS IN ('UNMATCHED', 'AMBIGUOUS')
                ) UNKNOWN_OR_AMBIGUOUS_SITE_TYPE_ECGIS
            FROM ({query}) q
            GROUP BY GATE_MISMATCH_DAYS
            ORDER BY GATE_MISMATCH_DAYS
        ) TO {sql_path(distribution_path)} (FORMAT CSV, HEADER TRUE)
    """)
    counts = {
        str(status): int(count)
        for status, count in con.execute(f"""
            SELECT SITE_TYPE_MATCH_STATUS, COUNT(*)
            FROM ({query}) q
            GROUP BY SITE_TYPE_MATCH_STATUS
        """).fetchall()
    }
    covered = sum(counts.values())
    if covered != int(run_info.candidate_count):
        raise AssertionError(
            "Streaming SITE_TYPE audit did not cover every gated candidate ECGI."
        )
    matched = sum(
        count for status, count in counts.items()
        if status not in {"UNMATCHED", "AMBIGUOUS"}
    )
    run_info.site_type_match_ratio = matched / max(1, int(run_info.candidate_count))
    cow_count = int(con.execute(f"""
        SELECT COUNT(*) FROM ({query}) q WHERE IS_COW_SITE=1
    """).fetchone()[0] or 0)
    log(
        f"Candidate SITE_TYPE coverage={run_info.site_type_match_ratio:.1%}; "
        f"COW candidates={cow_count:,}; retained={covered:,}; dropped=0; "
        f"status={counts}.",
        1,
    )
    return counts


def load_candidate_site_context_subset(
    con,
    ecgis: Sequence[object],
) -> pd.DataFrame:
    """Load context only for candidate ECGIs that survived evidence preloading."""
    keys = sorted({norm_key(value) for value in ecgis if norm_key(value)})
    columns = [
        "ECGI", "SITE_TYPE", "SITE_TYPE_MATCH_STATUS", "IS_COW_SITE",
        "SITE_TYPE_SOURCE_ROW_COUNT", "SITE_TYPE_DISTINCT_COUNT",
    ]
    if not keys:
        return pd.DataFrame(columns=columns)
    key_frame = pd.DataFrame({"ECGI": keys})
    con.register("active_context_ecgis_df", key_frame)
    try:
        context = con.execute(
            candidate_site_context_sql("active_context_ecgis_df")
            + " ORDER BY ECGI"
        ).fetchdf()
    finally:
        try:
            con.unregister("active_context_ecgis_df")
        except Exception:
            pass
    context.columns = [clean(column) for column in context.columns]
    context["ECGI"] = context["ECGI"].map(norm_key)
    if context["ECGI"].duplicated().any():
        raise AssertionError("Active SITE_TYPE context contains duplicate ECGIs.")
    return context


def build_populations(
    con,
    input_path: Path,
    colmap: Dict[str, str],
    date_column: str,
    args: argparse.Namespace,
) -> RunInfo:
    gate_mode = str(getattr(args, "candidate_gate", "window_any_mismatch"))
    gate_days = int(getattr(args, "candidate_gate_days", 0) or args.days)
    header(
        "PHASE 1 - "
        + (
            "ANY-WINDOW MISMATCH CANDIDATES"
            if gate_mode == "window_any_mismatch"
            else "LATEST-DAY MISMATCH CANDIDATES"
        )
        + " AND REPRESENTATIVE NORMAL SAMPLE"
    )
    if "DB_MISMATCH" not in colmap:
        raise ValueError("DB_MISMATCH is required for the selected candidate gate.")
    source = sql_path(input_path)
    date_sql = date_expr(colmap[date_column])
    ecgi = ecgi_sql(colmap)
    mismatch = compact_expr(quote_ident(colmap["DB_MISMATCH"]))
    stats = con.execute(f"""
        WITH raw AS (
            SELECT {date_sql} LOAD_DATE
            FROM read_csv_auto({source}, HEADER=TRUE, ALL_VARCHAR=TRUE, SAMPLE_SIZE=100000)
        )
        SELECT MIN(LOAD_DATE) MIN_DATE, MAX(LOAD_DATE) MAX_DATE,
               COUNT(DISTINCT LOAD_DATE) DISTINCT_DAYS
        FROM raw WHERE LOAD_DATE IS NOT NULL
    """).fetchdf().iloc[0]
    if pd.isna(stats["MAX_DATE"]):
        raise RuntimeError("No valid CESMLC load date could be parsed.")
    raw_max = pd.to_datetime(stats["MAX_DATE"]).normalize()
    window_end = pd.to_datetime(args.end_date).normalize() if args.end_date else raw_max
    window_start = window_end - pd.Timedelta(days=int(args.days) - 1)
    gate_window_start = window_end - pd.Timedelta(days=gate_days - 1)
    stable_start = window_end - pd.Timedelta(days=int(args.stable_days) - 1)

    snapshot = con.execute(f"""
        WITH raw AS (
            SELECT {date_sql} LOAD_DATE, {ecgi} ECGI
            FROM read_csv_auto(
                {source}, HEADER=TRUE, ALL_VARCHAR=TRUE, SAMPLE_SIZE=100000
            )
            WHERE {date_sql} BETWEEN
                  DATE {sql_string(str((window_end - pd.Timedelta(days=7)).date()))}
                  AND DATE {sql_string(str(window_end.date()))}
        ), daily AS (
            SELECT LOAD_DATE, COUNT(DISTINCT ECGI) N
            FROM raw
            WHERE LOAD_DATE IS NOT NULL AND ECGI IS NOT NULL AND ECGI<>''
            GROUP BY LOAD_DATE
        )
        SELECT
            MAX(N) FILTER (WHERE LOAD_DATE=DATE {sql_string(str(window_end.date()))}) LATEST_N,
            MEDIAN(N) FILTER (WHERE LOAD_DATE<DATE {sql_string(str(window_end.date()))}) PRIOR_MEDIAN
        FROM daily
    """).fetchdf().iloc[0]
    latest_distinct_ecgis = scalar_int(snapshot.get("LATEST_N"), 0)
    if latest_distinct_ecgis == 0:
        raise RuntimeError(
            f"No CESMLC snapshot exists on analysis end date {window_end.date()}. "
            "Choose a populated --end-date or omit it to use the latest parsed date."
        )
    prior_median = scalar_float(snapshot.get("PRIOR_MEDIAN"), latest_distinct_ecgis)
    latest_snapshot_ratio = latest_distinct_ecgis / max(1.0, prior_median)
    if latest_snapshot_ratio < float(args.min_latest_snapshot_ratio):
        message = (
            f"Latest snapshot has {latest_distinct_ecgis:,} distinct ECGIs, only "
            f"{latest_snapshot_ratio:.1%} of the prior-day median."
        )
        if not args.allow_incomplete_latest:
            raise RuntimeError(
                message
                + " Use --allow-incomplete-latest only after verifying that the partial load is intentional."
            )
        log("WARNING: " + message, 1)

    meta_select = [
        f"{compact_expr(quote_ident(colmap[name]))} AS {quote_ident(name)}"
        if name in colmap else f"NULL::VARCHAR AS {quote_ident(name)}"
        for name in META_COLS
    ]
    context_names = [name for name in ID_CONTEXT_COLS if name in colmap]
    context_select = [
        f"{compact_expr(quote_ident(colmap[name]))} AS {quote_ident(name)}"
        for name in context_names
    ]
    server_filter = (
        f"AND SERVER_ID={sql_string(args.server_id.strip().upper())}" if args.server_id else ""
    )
    con.execute("DROP TABLE IF EXISTS latest_population_raw")
    con.execute(f"""
        CREATE TEMPORARY TABLE latest_population_raw AS
        SELECT {date_sql} LOAD_DATE, {ecgi} ECGI,
               {', '.join(meta_select + context_select)}, {mismatch} DB_MISMATCH
        FROM read_csv_auto({source}, HEADER=TRUE, ALL_VARCHAR=TRUE, SAMPLE_SIZE=100000)
        WHERE {date_sql}=DATE {sql_string(str(window_end.date()))}
          AND {ecgi} IS NOT NULL AND {ecgi}<>''
    """)
    aggregate_parts = [
        "ECGI",
        *[f"MAX({quote_ident(name)}) AS {quote_ident(name)}" for name in META_COLS],
        *[f"MAX({quote_ident(name)}) AS {quote_ident(name)}" for name in context_names],
        "COUNT(*) LATEST_RAW_ROWS",
    ]
    con.execute("DROP TABLE IF EXISTS candidate_ecgis")
    if gate_mode == "latest_mismatch":
        con.execute(f"""
            CREATE TABLE candidate_ecgis AS
            SELECT {', '.join(aggregate_parts)},
                   COUNT(*) GATE_MISMATCH_ROWS,
                   1::BIGINT GATE_MISMATCH_DAYS,
                   DATE {sql_string(str(window_end.date()))} GATE_FIRST_MISMATCH_DATE,
                   DATE {sql_string(str(window_end.date()))} GATE_LAST_MISMATCH_DATE,
                   'LATEST_MISMATCH' CANDIDATE_GATE_REASON,
                   'CANDIDATE' POPULATION
            FROM latest_population_raw
            WHERE DB_MISMATCH='YES' {server_filter}
            GROUP BY ECGI
        """)
    elif gate_mode == "window_any_mismatch":
        minimum_mismatch_days = int(args.candidate_mismatch_min_days)
        # This scan selects only mismatch rows before grouping. It does not
        # materialize the complete 2M-ECGI-by-31-day population in pandas.
        con.execute(f"""
            CREATE TABLE candidate_ecgis AS
            WITH raw AS (
                SELECT {date_sql} LOAD_DATE, {ecgi} ECGI,
                       {', '.join(meta_select + context_select)},
                       {mismatch} DB_MISMATCH
                FROM read_csv_auto(
                    {source}, HEADER=TRUE, ALL_VARCHAR=TRUE, SAMPLE_SIZE=100000
                )
                WHERE {date_sql} BETWEEN
                      DATE {sql_string(str(gate_window_start.date()))}
                      AND DATE {sql_string(str(window_end.date()))}
                  AND {ecgi} IS NOT NULL AND {ecgi}<>''
            ), mismatch_rows AS (
                SELECT * FROM raw
                WHERE DB_MISMATCH='YES' {server_filter}
            )
            SELECT
                ECGI,
                {', '.join(
                    [f"MAX({quote_ident(name)}) AS {quote_ident(name)}" for name in META_COLS]
                    + [f"MAX({quote_ident(name)}) AS {quote_ident(name)}" for name in context_names]
                )},
                COUNT(*) FILTER(
                    WHERE LOAD_DATE=DATE {sql_string(str(window_end.date()))}
                ) LATEST_RAW_ROWS,
                COUNT(*) GATE_MISMATCH_ROWS,
                COUNT(DISTINCT LOAD_DATE) GATE_MISMATCH_DAYS,
                MIN(LOAD_DATE) GATE_FIRST_MISMATCH_DATE,
                MAX(LOAD_DATE) GATE_LAST_MISMATCH_DATE,
                'WINDOW_ANY_MISMATCH' CANDIDATE_GATE_REASON,
                'CANDIDATE' POPULATION
            FROM mismatch_rows
            GROUP BY ECGI
            HAVING COUNT(DISTINCT LOAD_DATE)>={minimum_mismatch_days}
        """)
    else:
        raise ValueError(f"Unsupported --candidate-gate: {gate_mode}")
    con.execute("DROP TABLE IF EXISTS normal_baseline_ecgis")
    con.execute(f"""
        CREATE TABLE normal_baseline_ecgis AS
        WITH eligible AS (
            SELECT {', '.join(aggregate_parts)}
            FROM latest_population_raw
            WHERE COALESCE(DB_MISMATCH, '')<>'YES' {server_filter}
              AND ECGI NOT IN (SELECT ECGI FROM candidate_ecgis)
            GROUP BY ECGI
        ), sampled AS (
            SELECT * FROM eligible
            ORDER BY HASH(ECGI, {int(args.random_state)})
            LIMIT {max(0, int(args.normal_baseline_max_ecgis))}
        )
        SELECT *, 'BASELINE' POPULATION
        FROM sampled
    """)
    con.execute("DROP TABLE IF EXISTS population_ecgis")
    con.execute("""
        CREATE TABLE population_ecgis AS
        SELECT * FROM candidate_ecgis
        UNION ALL BY NAME
        SELECT * FROM normal_baseline_ecgis
    """)
    candidate_count = int(con.execute("SELECT COUNT(*) FROM candidate_ecgis").fetchone()[0] or 0)
    baseline_count = int(con.execute("SELECT COUNT(*) FROM normal_baseline_ecgis").fetchone()[0] or 0)
    latest_raw_rows = int(con.execute("SELECT COUNT(*) FROM latest_population_raw").fetchone()[0] or 0)
    if candidate_count == 0:
        log("No ECGI passed the selected mismatch gate. Empty outputs will be produced.", 1)

    def population_hash(table_name: str) -> str:
        # Avoid collecting/sorting up to two million identifiers in pandas.
        count, xor_hash = con.execute(f"""
            SELECT COUNT(*), COALESCE(BIT_XOR(HASH(ECGI)), 0)
            FROM {table_name}
        """).fetchone()
        payload = f"{table_name}|{int(count or 0)}|{int(xor_hash or 0)}"
        return hashlib.sha256(payload.encode()).hexdigest()[:16]

    candidate_hash = population_hash("candidate_ecgis")
    baseline_hash = population_hash("normal_baseline_ecgis")
    con.execute("DROP TABLE IF EXISTS latest_population_raw")
    log(
        f"Window={window_start.date()}..{window_end.date()}; gate={gate_mode}; "
        f"gate_window={gate_window_start.date()}..{window_end.date()}; "
        f"min_mismatch_days={int(args.candidate_mismatch_min_days)}; "
        f"candidates={candidate_count:,}; "
        f"baseline={baseline_count:,}; SITE_TYPE is not joined until after detection.",
        1,
    )
    return RunInfo(
        window_start=window_start,
        window_end=window_end,
        stable_start=stable_start,
        candidate_count=candidate_count,
        baseline_count=baseline_count,
        candidate_hash=candidate_hash,
        baseline_hash=baseline_hash,
        latest_raw_rows=latest_raw_rows,
        latest_distinct_ecgis=latest_distinct_ecgis,
        latest_snapshot_ratio=latest_snapshot_ratio,
        site_type_match_ratio=0.0,
    )


def create_history_tables(con) -> None:
    con.execute("DROP TABLE IF EXISTS feature_history_metrics")
    con.execute("""
        CREATE TABLE feature_history_metrics(
            POPULATION VARCHAR,
            ECGI VARCHAR,
            FEATURE VARCHAR,
            OBSERVED_DAYS BIGINT,
            FIRST_OBSERVED_DATE DATE,
            LAST_OBSERVED_DATE DATE,
            FIRST_SOURCE_VALUE VARCHAR,
            LATEST_SOURCE_VALUE VARCHAR,
            PREVIOUS_DIFFERENT_SOURCE_VALUE VARCHAR,
            DISTINCT_SOURCE_VALUES BIGINT,
            MODE_SOURCE_VALUE VARCHAR,
            MODE_SOURCE_VALUE_DAYS BIGINT,
            MODE_SOURCE_VALUE_SHARE_PERCENT DOUBLE,
            LATEST_SOURCE_VALUE_DAYS BIGINT,
            RECENT_OBSERVED_DAYS BIGINT,
            LATEST_OBSERVED_ON_END_DATE_FLAG INTEGER,
            RAW_ROWS_FOR_FEATURE BIGINT,
            LEGACY_MIN_CHANGE_EVENTS BIGINT,
            LEGACY_MIN_CHANGE_DAYS BIGINT,
            LEGACY_MIN_ONLY_CHANGE_DAYS BIGINT,
            MODAL_ONLY_CHANGE_DAYS BIGINT,
            MODAL_AND_LEGACY_CHANGE_DAYS BIGINT,
            DAILY_REPRESENTATIVE_DISAGREEMENT_DAYS BIGINT
        )
    """)
    con.execute("DROP TABLE IF EXISTS change_events")
    con.execute("""
        CREATE TABLE change_events(
            POPULATION VARCHAR,
            ECGI VARCHAR,
            FEATURE VARCHAR,
            LOAD_DATE DATE,
            PREVIOUS_LOAD_DATE DATE,
            PREVIOUS_SOURCE_VALUE VARCHAR,
            CURRENT_SOURCE_VALUE VARCHAR,
            RAW_ROWS_PER_ECGI_DATE BIGINT,
            A_B_A_REVERT_FLAG INTEGER,
            BOUNDARY_ANCHOR_EVENT_FLAG INTEGER
        )
    """)
    con.execute("DROP TABLE IF EXISTS same_day_conflicts")
    con.execute("""
        CREATE TABLE same_day_conflicts(
            POPULATION VARCHAR,
            ECGI VARCHAR,
            FEATURE VARCHAR,
            LOAD_DATE DATE,
            REPRESENTATIVE_SOURCE_VALUE VARCHAR,
            SOURCE_VALUES_UNIQUE_COUNT BIGINT,
            SOURCE_VALUES_WITH_COUNTS VARCHAR,
            RAW_ROWS_PER_ECGI_DATE BIGINT
        )
    """)


def create_daily_modal_table(con, feature: str) -> None:
    """Build the decision value and a non-decision V9.6 reconciliation value.

    The modal value is the only representation used by detection.  V9.6 used
    lexical MIN(value), which lets one minority duplicate row replace the
    majority same-day value.  Keeping that value beside the mode lets every
    production run quantify the old/new semantic gap without restoring the
    fragile legacy representative to anomaly decisions.
    """
    q = quote_ident(feature)
    con.execute("DROP TABLE IF EXISTS feature_daily")
    con.execute(f"""
        CREATE TEMPORARY TABLE feature_daily AS
        WITH value_counts AS (
            SELECT POPULATION, ECGI, LOAD_DATE, {q} SOURCE_VALUE, COUNT(*) VALUE_RAW_ROWS
            FROM raw_population_batch
            WHERE {q} IS NOT NULL
            GROUP BY POPULATION, ECGI, LOAD_DATE, {q}
        ), daily AS (
            SELECT
                POPULATION,
                ECGI,
                LOAD_DATE,
                FIRST(SOURCE_VALUE ORDER BY VALUE_RAW_ROWS DESC, SOURCE_VALUE ASC) SOURCE_VALUE,
                MIN(SOURCE_VALUE) LEGACY_MIN_SOURCE_VALUE,
                SUM(VALUE_RAW_ROWS) RAW_ROWS_PER_ECGI_DATE,
                COUNT(*) SOURCE_VALUES_UNIQUE_COUNT
            FROM value_counts
            GROUP BY POPULATION, ECGI, LOAD_DATE
        ), conflict_text AS (
            SELECT v.POPULATION, v.ECGI, v.LOAD_DATE,
                   STRING_AGG(
                       v.SOURCE_VALUE || ' (' || CAST(v.VALUE_RAW_ROWS AS VARCHAR) || ')',
                       ' | ' ORDER BY v.VALUE_RAW_ROWS DESC, v.SOURCE_VALUE ASC
                   ) SOURCE_VALUES_WITH_COUNTS
            FROM value_counts v
            JOIN daily d USING(POPULATION, ECGI, LOAD_DATE)
            WHERE d.SOURCE_VALUES_UNIQUE_COUNT>1
            GROUP BY v.POPULATION, v.ECGI, v.LOAD_DATE
        )
        SELECT d.*,
               CASE WHEN d.SOURCE_VALUES_UNIQUE_COUNT=1
                    THEN d.SOURCE_VALUE || ' (' || CAST(d.RAW_ROWS_PER_ECGI_DATE AS VARCHAR) || ')'
                    ELSE c.SOURCE_VALUES_WITH_COUNTS END SOURCE_VALUES_WITH_COUNTS
        FROM daily d
        LEFT JOIN conflict_text c USING(POPULATION, ECGI, LOAD_DATE)
    """)


def insert_feature_history(con, feature: str, run_info: RunInfo) -> None:
    fsql = sql_string(feature)
    start = sql_string(str(run_info.window_start.date()))
    recent = sql_string(str(run_info.stable_start.date()))
    end = sql_string(str(run_info.window_end.date()))
    con.execute(f"""
        INSERT INTO feature_history_metrics
        WITH history_daily AS (
            SELECT * FROM feature_daily WHERE SOURCE_VALUE IS NOT NULL
        ), daily AS (
            SELECT * FROM history_daily
            WHERE LOAD_DATE BETWEEN DATE {start} AND DATE {end}
        ), representation_sequence AS (
            SELECT
                POPULATION,
                ECGI,
                LOAD_DATE,
                SOURCE_VALUE,
                LEGACY_MIN_SOURCE_VALUE,
                LAG(SOURCE_VALUE) OVER(
                    PARTITION BY POPULATION, ECGI ORDER BY LOAD_DATE
                ) PREVIOUS_MODAL_SOURCE_VALUE,
                LAG(LEGACY_MIN_SOURCE_VALUE) OVER(
                    PARTITION BY POPULATION, ECGI ORDER BY LOAD_DATE
                ) PREVIOUS_LEGACY_MIN_SOURCE_VALUE
            FROM history_daily
        ), representation_summary AS (
            SELECT
                POPULATION,
                ECGI,
                SUM(CASE
                    WHEN PREVIOUS_LEGACY_MIN_SOURCE_VALUE IS NOT NULL
                     AND LEGACY_MIN_SOURCE_VALUE<>PREVIOUS_LEGACY_MIN_SOURCE_VALUE
                    THEN 1 ELSE 0 END) LEGACY_MIN_CHANGE_EVENTS,
                COUNT(DISTINCT CASE
                    WHEN PREVIOUS_LEGACY_MIN_SOURCE_VALUE IS NOT NULL
                     AND LEGACY_MIN_SOURCE_VALUE<>PREVIOUS_LEGACY_MIN_SOURCE_VALUE
                    THEN LOAD_DATE END) LEGACY_MIN_CHANGE_DAYS,
                COUNT(DISTINCT CASE
                    WHEN PREVIOUS_LEGACY_MIN_SOURCE_VALUE IS NOT NULL
                     AND LEGACY_MIN_SOURCE_VALUE<>PREVIOUS_LEGACY_MIN_SOURCE_VALUE
                     AND NOT (
                         PREVIOUS_MODAL_SOURCE_VALUE IS NOT NULL
                         AND SOURCE_VALUE<>PREVIOUS_MODAL_SOURCE_VALUE
                     )
                    THEN LOAD_DATE END) LEGACY_MIN_ONLY_CHANGE_DAYS,
                COUNT(DISTINCT CASE
                    WHEN PREVIOUS_MODAL_SOURCE_VALUE IS NOT NULL
                     AND SOURCE_VALUE<>PREVIOUS_MODAL_SOURCE_VALUE
                     AND NOT (
                         PREVIOUS_LEGACY_MIN_SOURCE_VALUE IS NOT NULL
                         AND LEGACY_MIN_SOURCE_VALUE<>PREVIOUS_LEGACY_MIN_SOURCE_VALUE
                     )
                    THEN LOAD_DATE END) MODAL_ONLY_CHANGE_DAYS,
                COUNT(DISTINCT CASE
                    WHEN PREVIOUS_MODAL_SOURCE_VALUE IS NOT NULL
                     AND SOURCE_VALUE<>PREVIOUS_MODAL_SOURCE_VALUE
                     AND PREVIOUS_LEGACY_MIN_SOURCE_VALUE IS NOT NULL
                     AND LEGACY_MIN_SOURCE_VALUE<>PREVIOUS_LEGACY_MIN_SOURCE_VALUE
                    THEN LOAD_DATE END) MODAL_AND_LEGACY_CHANGE_DAYS,
                COUNT(DISTINCT CASE
                    WHEN SOURCE_VALUE<>LEGACY_MIN_SOURCE_VALUE THEN LOAD_DATE
                    END) DAILY_REPRESENTATIVE_DISAGREEMENT_DAYS
            FROM representation_sequence
            WHERE LOAD_DATE BETWEEN DATE {start} AND DATE {end}
            GROUP BY POPULATION, ECGI
        ), aggregate_base AS (
            SELECT POPULATION, ECGI,
                   COUNT(*) OBSERVED_DAYS,
                   MIN(LOAD_DATE) FIRST_OBSERVED_DATE,
                   MAX(LOAD_DATE) LAST_OBSERVED_DATE,
                   COUNT(DISTINCT SOURCE_VALUE) DISTINCT_SOURCE_VALUES,
                   COUNT(*) FILTER (WHERE LOAD_DATE>=DATE {recent}) RECENT_OBSERVED_DAYS,
                   SUM(RAW_ROWS_PER_ECGI_DATE) RAW_ROWS_FOR_FEATURE
            FROM daily GROUP BY POPULATION, ECGI
        ), first_value AS (
            SELECT POPULATION, ECGI, SOURCE_VALUE FIRST_SOURCE_VALUE
            FROM (
                SELECT *, ROW_NUMBER() OVER(
                    PARTITION BY POPULATION, ECGI ORDER BY LOAD_DATE ASC, SOURCE_VALUE ASC
                ) RN FROM daily
            ) WHERE RN=1
        ), latest_value AS (
            SELECT POPULATION, ECGI, SOURCE_VALUE LATEST_SOURCE_VALUE
            FROM (
                SELECT *, ROW_NUMBER() OVER(
                    PARTITION BY POPULATION, ECGI ORDER BY LOAD_DATE DESC, SOURCE_VALUE ASC
                ) RN FROM daily
            ) WHERE RN=1
        ), previous_different AS (
            SELECT POPULATION, ECGI, SOURCE_VALUE PREVIOUS_DIFFERENT_SOURCE_VALUE
            FROM (
                SELECT d.*, ROW_NUMBER() OVER(
                    PARTITION BY d.POPULATION, d.ECGI
                    ORDER BY d.LOAD_DATE DESC, d.SOURCE_VALUE ASC
                ) RN
                FROM history_daily d JOIN latest_value l USING(POPULATION, ECGI)
                WHERE d.SOURCE_VALUE<>l.LATEST_SOURCE_VALUE
            ) WHERE RN=1
        ), value_days AS (
            SELECT POPULATION, ECGI, SOURCE_VALUE, COUNT(*) VALUE_DAYS
            FROM daily GROUP BY POPULATION, ECGI, SOURCE_VALUE
        ), mode_value AS (
            SELECT POPULATION, ECGI, SOURCE_VALUE MODE_SOURCE_VALUE,
                   VALUE_DAYS MODE_SOURCE_VALUE_DAYS
            FROM (
                SELECT *, ROW_NUMBER() OVER(
                    PARTITION BY POPULATION, ECGI ORDER BY VALUE_DAYS DESC, SOURCE_VALUE ASC
                ) RN FROM value_days
            ) WHERE RN=1
        ), latest_counts AS (
            SELECT v.POPULATION, v.ECGI, v.VALUE_DAYS LATEST_SOURCE_VALUE_DAYS
            FROM value_days v JOIN latest_value l
              ON v.POPULATION=l.POPULATION AND v.ECGI=l.ECGI
             AND v.SOURCE_VALUE=l.LATEST_SOURCE_VALUE
        )
        SELECT
            a.POPULATION, a.ECGI, {fsql}, a.OBSERVED_DAYS,
            a.FIRST_OBSERVED_DATE, a.LAST_OBSERVED_DATE,
            f.FIRST_SOURCE_VALUE, l.LATEST_SOURCE_VALUE,
            p.PREVIOUS_DIFFERENT_SOURCE_VALUE,
            a.DISTINCT_SOURCE_VALUES,
            m.MODE_SOURCE_VALUE, m.MODE_SOURCE_VALUE_DAYS,
            100.0*m.MODE_SOURCE_VALUE_DAYS/NULLIF(a.OBSERVED_DAYS, 0),
            lc.LATEST_SOURCE_VALUE_DAYS,
            a.RECENT_OBSERVED_DAYS,
            CASE WHEN a.LAST_OBSERVED_DATE=DATE {end} THEN 1 ELSE 0 END,
            a.RAW_ROWS_FOR_FEATURE,
            COALESCE(r.LEGACY_MIN_CHANGE_EVENTS, 0),
            COALESCE(r.LEGACY_MIN_CHANGE_DAYS, 0),
            COALESCE(r.LEGACY_MIN_ONLY_CHANGE_DAYS, 0),
            COALESCE(r.MODAL_ONLY_CHANGE_DAYS, 0),
            COALESCE(r.MODAL_AND_LEGACY_CHANGE_DAYS, 0),
            COALESCE(r.DAILY_REPRESENTATIVE_DISAGREEMENT_DAYS, 0)
        FROM aggregate_base a
        LEFT JOIN first_value f USING(POPULATION, ECGI)
        LEFT JOIN latest_value l USING(POPULATION, ECGI)
        LEFT JOIN previous_different p USING(POPULATION, ECGI)
        LEFT JOIN mode_value m USING(POPULATION, ECGI)
        LEFT JOIN latest_counts lc USING(POPULATION, ECGI)
        LEFT JOIN representation_summary r USING(POPULATION, ECGI)
    """)
    con.execute(f"""
        INSERT INTO change_events
        WITH sequence AS (
            SELECT POPULATION, ECGI, LOAD_DATE, SOURCE_VALUE,
                   LAG(LOAD_DATE) OVER(
                       PARTITION BY POPULATION, ECGI ORDER BY LOAD_DATE
                   ) PREVIOUS_LOAD_DATE,
                   LAG(SOURCE_VALUE) OVER(
                       PARTITION BY POPULATION, ECGI ORDER BY LOAD_DATE
                   ) PREVIOUS_SOURCE_VALUE,
                   LAG(SOURCE_VALUE, 2) OVER(
                       PARTITION BY POPULATION, ECGI ORDER BY LOAD_DATE
                   ) TWO_BACK_SOURCE_VALUE,
                   RAW_ROWS_PER_ECGI_DATE
            FROM feature_daily
        )
        SELECT POPULATION, ECGI, {fsql}, LOAD_DATE, PREVIOUS_LOAD_DATE,
               PREVIOUS_SOURCE_VALUE, SOURCE_VALUE, RAW_ROWS_PER_ECGI_DATE,
               CASE WHEN TWO_BACK_SOURCE_VALUE IS NOT NULL
                      AND SOURCE_VALUE=TWO_BACK_SOURCE_VALUE
                      AND SOURCE_VALUE<>PREVIOUS_SOURCE_VALUE
                    THEN 1 ELSE 0 END,
               CASE WHEN PREVIOUS_LOAD_DATE<DATE {start} THEN 1 ELSE 0 END
        FROM sequence
        WHERE LOAD_DATE BETWEEN DATE {start} AND DATE {end}
          AND PREVIOUS_SOURCE_VALUE IS NOT NULL
          AND SOURCE_VALUE<>PREVIOUS_SOURCE_VALUE
    """)
    con.execute(f"""
        INSERT INTO same_day_conflicts
        SELECT POPULATION, ECGI, {fsql}, LOAD_DATE, SOURCE_VALUE,
               SOURCE_VALUES_UNIQUE_COUNT, SOURCE_VALUES_WITH_COUNTS,
               RAW_ROWS_PER_ECGI_DATE
        FROM feature_daily
        WHERE LOAD_DATE BETWEEN DATE {start} AND DATE {end}
          AND SOURCE_VALUES_UNIQUE_COUNT>1
    """)


def build_history_tables(
    con,
    input_path: Path,
    colmap: Dict[str, str],
    date_column: str,
    features: List[str],
    run_info: RunInfo,
    args: argparse.Namespace,
) -> None:
    header("PHASE 2 - IDENTICAL CANDIDATE/NORMAL SOURCE-HISTORY ENGINEERING")
    create_history_tables(con)
    if run_info.candidate_count == 0 and run_info.baseline_count == 0:
        return
    source = sql_path(input_path)
    date_sql = date_expr(colmap[date_column])
    ecgi = ecgi_sql(colmap)
    start = sql_string(str(run_info.window_start.date()))
    anchor_start = sql_string(str(
        (run_info.window_start - pd.Timedelta(days=int(args.history_anchor_days))).date()
    ))
    end = sql_string(str(run_info.window_end.date()))
    batches = list(chunks(features, int(args.feature_batch_size)))
    for batch_number, batch in enumerate(batches, 1):
        log(f"Feature batch {batch_number}/{len(batches)}: {', '.join(batch)}", 1)
        selections = [
            f"{feature_expr(colmap[feature], feature, args.latlon_decimals)} "
            f"AS {quote_ident(feature)}"
            for feature in batch
        ]
        con.execute("DROP TABLE IF EXISTS raw_population_batch")
        con.execute(f"""
            CREATE TEMPORARY TABLE raw_population_batch AS
            WITH raw AS (
                SELECT {date_sql} LOAD_DATE, {ecgi} ECGI, {', '.join(selections)}
                FROM read_csv_auto(
                    {source}, HEADER=TRUE, ALL_VARCHAR=TRUE, SAMPLE_SIZE=100000
                )
            )
            SELECT p.POPULATION, r.*
            FROM raw r JOIN population_ecgis p USING(ECGI)
            WHERE r.LOAD_DATE BETWEEN DATE {anchor_start} AND DATE {end}
              AND r.ECGI IS NOT NULL AND r.ECGI<>''
        """)
        for feature in batch:
            create_daily_modal_table(con, feature)
            insert_feature_history(con, feature, run_info)
        metric_rows = int(con.execute("SELECT COUNT(*) FROM feature_history_metrics").fetchone()[0] or 0)
        event_rows = int(con.execute("SELECT COUNT(*) FROM change_events").fetchone()[0] or 0)
        log(f"Cumulative metrics={metric_rows:,}; source-change events={event_rows:,}", 2)
    con.execute("DROP TABLE IF EXISTS feature_daily")
    con.execute("DROP TABLE IF EXISTS raw_population_batch")


def load_history_tables(con) -> Tuple[pd.DataFrame, ...]:
    """Load all normal rows but only candidate features with real evidence.

    Stable candidate-feature rows cannot qualify for any of the four clusters.
    Keeping those rows in DuckDB prevents an any-window gate from turning into
    a multi-million-row pandas allocation. Inter-day changes and same-day
    conflicts are both retained, so anomaly recall is unchanged.
    """
    con.execute("DROP TABLE IF EXISTS active_candidate_feature_keys")
    con.execute("""
        CREATE TEMPORARY TABLE active_candidate_feature_keys AS
        SELECT DISTINCT POPULATION, ECGI, FEATURE
        FROM change_events
        WHERE POPULATION='CANDIDATE'
        UNION
        SELECT DISTINCT POPULATION, ECGI, FEATURE
        FROM same_day_conflicts
        WHERE POPULATION='CANDIDATE'
    """)
    con.execute("DROP TABLE IF EXISTS loaded_feature_history_metrics")
    con.execute("""
        CREATE TEMPORARY TABLE loaded_feature_history_metrics AS
        SELECT m.*
        FROM feature_history_metrics m
        WHERE m.POPULATION='BASELINE'
        UNION ALL
        SELECT m.*
        FROM feature_history_metrics m
        INNER JOIN active_candidate_feature_keys k
          ON m.POPULATION=k.POPULATION
         AND m.ECGI=k.ECGI
         AND m.FEATURE=k.FEATURE
        WHERE m.POPULATION='CANDIDATE'
    """)
    con.execute("DROP TABLE IF EXISTS loaded_population_keys")
    con.execute("""
        CREATE TEMPORARY TABLE loaded_population_keys AS
        SELECT DISTINCT POPULATION, ECGI
        FROM loaded_feature_history_metrics
    """)
    all_candidate_count = int(
        con.execute("SELECT COUNT(*) FROM candidate_ecgis").fetchone()[0] or 0
    )
    active_candidate_count = int(con.execute("""
        SELECT COUNT(*) FROM loaded_population_keys
        WHERE POPULATION='CANDIDATE'
    """).fetchone()[0] or 0)
    log(
        f"Memory prefilter retained {active_candidate_count:,}/{all_candidate_count:,} "
        "candidate ECGIs with an inter-day change or same-day conflict; "
        f"stable/no-conflict candidates kept out of pandas={max(0, all_candidate_count-active_candidate_count):,}.",
        1,
    )
    frames = (
        con.execute("""
            SELECT c.*
            FROM candidate_ecgis c
            INNER JOIN loaded_population_keys k
              ON c.ECGI=k.ECGI AND k.POPULATION='CANDIDATE'
        """).fetchdf(),
        con.execute("""
            SELECT p.*
            FROM population_ecgis p
            INNER JOIN loaded_population_keys k
              ON p.POPULATION=k.POPULATION AND p.ECGI=k.ECGI
        """).fetchdf(),
        con.execute("SELECT * FROM loaded_feature_history_metrics").fetchdf(),
        con.execute("""
            SELECT e.*
            FROM change_events e
            INNER JOIN loaded_population_keys k
              ON e.POPULATION=k.POPULATION AND e.ECGI=k.ECGI
        """).fetchdf(),
        con.execute("""
            SELECT c.*
            FROM same_day_conflicts c
            INNER JOIN loaded_population_keys k
              ON c.POPULATION=k.POPULATION AND c.ECGI=k.ECGI
        """).fetchdf(),
    )
    for frame in frames:
        frame.columns = [clean(column) for column in frame.columns]
        if "ECGI" in frame.columns:
            frame["ECGI"] = frame["ECGI"].map(norm_key)
        if "FEATURE" in frame.columns:
            frame["FEATURE"] = frame["FEATURE"].map(clean)
        for column in [
            "LOAD_DATE", "PREVIOUS_LOAD_DATE", "FIRST_OBSERVED_DATE",
            "LAST_OBSERVED_DATE",
        ]:
            if column in frame.columns:
                frame[column] = pd.to_datetime(frame[column], errors="coerce").dt.normalize()
        for column in [
            "FIRST_SOURCE_VALUE", "LATEST_SOURCE_VALUE",
            "PREVIOUS_DIFFERENT_SOURCE_VALUE", "MODE_SOURCE_VALUE",
            "PREVIOUS_SOURCE_VALUE", "CURRENT_SOURCE_VALUE",
            "REPRESENTATIVE_SOURCE_VALUE",
        ]:
            if column in frame.columns:
                frame[column] = frame[column].map(norm_val)
    return frames


def cache_manifest(
    input_path: Path,
    features: List[str],
    run_info: RunInfo,
    args: argparse.Namespace,
) -> Dict[str, Any]:
    return {
        "cache_schema": CACHE_SCHEMA,
        "input_path": str(input_path.resolve()),
        "input_basename": input_path.name,
        "input_fingerprint": (
            full_file_fingerprint(input_path)
            if bool(getattr(args, "reuse_history_cache", False))
            else "CACHE_DISABLED"
        ),
        "features": sorted(features),
        "window_start": str(run_info.window_start.date()),
        "window_end": str(run_info.window_end.date()),
        "stable_window_start": str(run_info.stable_start.date()),
        "stable_days": int(args.stable_days),
        "history_anchor_days": int(args.history_anchor_days),
        "candidate_gate": str(args.candidate_gate),
        "candidate_gate_days": int(args.candidate_gate_days or args.days),
        "candidate_mismatch_min_days": int(args.candidate_mismatch_min_days),
        "candidate_pandas_loading_policy": (
            "baseline-all;candidate-interday-change-or-sameday-conflict-only"
        ),
        "candidate_hash": run_info.candidate_hash,
        "baseline_hash": run_info.baseline_hash,
        "latlon_decimals": int(args.latlon_decimals),
        "daily_policy": (
            "daily-mode-then-lexical-tie-break-with-pre-window-anchor;"
            "legacy-lexical-min-retained-for-reconciliation-only"
        ),
        "boundary_event_policy": "lag-with-anchor-retain-current-date-inside-analysis-window",
        "physicalcellid_policy": "excluded-before-history",
        "site_type_policy": "attached-after-detection-context-only",
    }


def cache_compare(manifest: Dict[str, Any], mode: str, identity: str) -> Dict[str, Any]:
    mode = str(mode).lower()
    identity = str(identity).lower()
    keys = [
        "cache_schema", "features", "window_start", "window_end",
        "stable_window_start", "stable_days", "history_anchor_days",
        "candidate_gate", "candidate_gate_days", "candidate_mismatch_min_days",
        "candidate_pandas_loading_policy",
        "candidate_hash", "baseline_hash", "latlon_decimals", "daily_policy",
        "boundary_event_policy", "physicalcellid_policy", "site_type_policy",
    ]
    if mode == "strict":
        keys += ["input_path", "input_basename", "input_fingerprint"]
    elif mode == "compatible":
        # Compatible still checks bytes.  Basename controls path portability only.
        keys += ["input_fingerprint"]
        if identity == "exact":
            keys += ["input_path"]
        elif identity == "basename":
            keys += ["input_basename"]
    return {key: manifest.get(key) for key in sorted(set(keys))}


def load_cache(
    cache_dir: Path,
    expected: Dict[str, Any],
    args: argparse.Namespace,
) -> Optional[Tuple[pd.DataFrame, ...]]:
    names = [
        "v98_candidate_ecgis.pkl",
        "v98_population_ecgis.pkl",
        "v98_feature_history_metrics.pkl",
        "v98_change_events.pkl",
        "v98_same_day_conflicts.pkl",
    ]
    manifest_path = cache_dir / "v98_history_cache_manifest.json"
    if not manifest_path.exists() or any(not (cache_dir / name).exists() for name in names):
        return None
    try:
        saved = json.loads(manifest_path.read_text(encoding="utf-8"))["manifest"]
        cached_view = cache_compare(saved, args.cache_match_mode, args.cache_input_identity)
        expected_view = cache_compare(expected, args.cache_match_mode, args.cache_input_identity)
        if cached_view != expected_view:
            diagnostics = {
                "status": "cache_invalidated",
                "differing_keys": {
                    key: {"cached": cached_view.get(key), "expected": expected_view.get(key)}
                    for key in sorted(set(cached_view) | set(expected_view))
                    if cached_view.get(key) != expected_view.get(key)
                },
            }
            cache_dir.mkdir(parents=True, exist_ok=True)
            (cache_dir / "v98_cache_invalidation.json").write_text(
                json.dumps(diagnostics, indent=2, default=str), encoding="utf-8"
            )
            return None
        frames = tuple(pd.read_pickle(cache_dir / name) for name in names)
        log(f"Loaded exact V9.8 history cache: {cache_dir}", 1)
        return frames
    except Exception as exc:
        log(f"V9.8 cache load failed ({exc}); rebuilding.", 1)
        return None


def save_cache(
    cache_dir: Path,
    manifest: Dict[str, Any],
    frames: Tuple[pd.DataFrame, ...],
) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    names = [
        "v98_candidate_ecgis.pkl",
        "v98_population_ecgis.pkl",
        "v98_feature_history_metrics.pkl",
        "v98_change_events.pkl",
        "v98_same_day_conflicts.pkl",
    ]
    for frame, name in zip(frames, names):
        frame.to_pickle(cache_dir / name)
    payload = {
        "manifest": manifest,
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "row_counts": {name: len(frame) for name, frame in zip(names, frames)},
    }
    (cache_dir / "v98_history_cache_manifest.json").write_text(
        json.dumps(payload, indent=2, default=str), encoding="utf-8"
    )
    log(f"Saved V9.8 history cache: {cache_dir}", 1)


def add_reverse_transition_flags(changes: pd.DataFrame) -> pd.DataFrame:
    if changes is None or changes.empty:
        result = pd.DataFrame() if changes is None else changes.copy()
        result["REVERSE_TRANSITION_FLAG"] = pd.Series(dtype=int)
        result["TRANSITION_EDGE"] = pd.Series(dtype=object)
        return result
    result = changes.sort_values(
        ["POPULATION", "ECGI", "FEATURE", "LOAD_DATE", "PREVIOUS_SOURCE_VALUE", "CURRENT_SOURCE_VALUE"]
    ).reset_index(drop=True).copy()
    result["TRANSITION_EDGE"] = (
        result["PREVIOUS_SOURCE_VALUE"].fillna("").astype(str)
        + "->" + result["CURRENT_SOURCE_VALUE"].fillna("").astype(str)
    )
    keys = ["POPULATION", "ECGI", "FEATURE"]
    reverse_first = (
        result.groupby(
            keys + ["PREVIOUS_SOURCE_VALUE", "CURRENT_SOURCE_VALUE"], as_index=False
        )["LOAD_DATE"].min()
        .rename(columns={
            "PREVIOUS_SOURCE_VALUE": "CURRENT_SOURCE_VALUE",
            "CURRENT_SOURCE_VALUE": "PREVIOUS_SOURCE_VALUE",
            "LOAD_DATE": "OPPOSITE_EDGE_FIRST_DATE",
        })
    )
    result = result.merge(
        reverse_first,
        on=keys + ["PREVIOUS_SOURCE_VALUE", "CURRENT_SOURCE_VALUE"],
        how="left",
    )
    result["REVERSE_TRANSITION_FLAG"] = (
        pd.to_datetime(result["OPPOSITE_EDGE_FIRST_DATE"], errors="coerce")
        < pd.to_datetime(result["LOAD_DATE"], errors="coerce")
    ).astype(int)
    result.drop(columns=["OPPOSITE_EDGE_FIRST_DATE"], inplace=True)
    return result


def build_change_statistics(
    changes: pd.DataFrame,
    run_info: RunInfo,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    keys = ["POPULATION", "ECGI", "FEATURE"]
    if changes is None or changes.empty:
        return pd.DataFrame(columns=keys), add_reverse_transition_flags(changes)
    events = add_reverse_transition_flags(changes)
    events["LOAD_DATE"] = pd.to_datetime(events["LOAD_DATE"], errors="coerce").dt.normalize()
    events["BOUNDARY_ANCHOR_EVENT_FLAG"] = col_num(
        events, "BOUNDARY_ANCHOR_EVENT_FLAG", 0
    ).astype(int)
    events = events.sort_values(keys + ["LOAD_DATE"]).reset_index(drop=True)
    events["IS_RECENT_STABILITY_WINDOW"] = events["LOAD_DATE"].ge(run_info.stable_start).astype(int)
    age = (run_info.window_end - events["LOAD_DATE"]).dt.days.clip(lower=0)
    events["RECENCY_WEIGHT"] = np.exp(-safe_num(age, 9999) / max(1.0, 0.7 * (
        (run_info.window_end - run_info.stable_start).days + 1
    )))
    events["CHANGE_INTERVAL_DAYS"] = events.groupby(keys)["LOAD_DATE"].diff().dt.days
    stats = events.groupby(keys, as_index=False).agg(
        SOURCE_CHANGE_EVENTS=("LOAD_DATE", "size"),
        SOURCE_CHANGE_DAYS=("LOAD_DATE", "nunique"),
        BOUNDARY_CHANGE_EVENTS_RECOVERED=("BOUNDARY_ANCHOR_EVENT_FLAG", "sum"),
        FIRST_CHANGE_DATE=("LOAD_DATE", "min"),
        LAST_CHANGE_DATE=("LOAD_DATE", "max"),
        A_B_A_REVERT_COUNT=("A_B_A_REVERT_FLAG", "sum"),
        REVERSE_TRANSITION_COUNT=("REVERSE_TRANSITION_FLAG", "sum"),
        UNIQUE_TRANSITION_EDGE_COUNT=("TRANSITION_EDGE", "nunique"),
        RECENCY_WEIGHTED_ACTIVITY=("RECENCY_WEIGHT", "sum"),
        MEAN_CHANGE_INTERVAL_DAYS=("CHANGE_INTERVAL_DAYS", "mean"),
        STD_CHANGE_INTERVAL_DAYS=("CHANGE_INTERVAL_DAYS", "std"),
    )
    recent = (
        events.loc[events["IS_RECENT_STABILITY_WINDOW"].eq(1)]
        .groupby(keys, as_index=False)["LOAD_DATE"].nunique()
        .rename(columns={"LOAD_DATE": "RECENT_CHANGE_DAYS"})
    )
    latest = (
        events.sort_values(keys + ["LOAD_DATE"])
        .groupby(keys, as_index=False).tail(1)[
            keys + ["LOAD_DATE", "PREVIOUS_SOURCE_VALUE", "CURRENT_SOURCE_VALUE", "TRANSITION_EDGE"]
        ]
        .rename(columns={
            "LOAD_DATE": "LATEST_CHANGE_DATE",
            "PREVIOUS_SOURCE_VALUE": "LATEST_PREVIOUS_SOURCE_VALUE",
            "CURRENT_SOURCE_VALUE": "LATEST_CURRENT_SOURCE_VALUE",
            "TRANSITION_EDGE": "LATEST_TRANSITION_EDGE",
        })
    )
    stats = stats.merge(recent, on=keys, how="left").merge(latest, on=keys, how="left")
    stats["RECENT_CHANGE_DAYS"] = col_num(stats, "RECENT_CHANGE_DAYS", 0)
    stats["DAYS_SINCE_LAST_CHANGE"] = (
        run_info.window_end
        - pd.to_datetime(stats["LAST_CHANGE_DATE"], errors="coerce").dt.normalize()
    ).dt.days.fillna(9999).clip(lower=0)
    event_count = col_num(stats, "SOURCE_CHANGE_EVENTS", 0).clip(lower=1)
    stats["ABA_EVENT_RATE"] = (
        col_num(stats, "A_B_A_REVERT_COUNT", 0) / event_count
    ).clip(0, 1)
    stats["REVERSE_EVENT_RATE"] = (
        col_num(stats, "REVERSE_TRANSITION_COUNT", 0) / event_count
    ).clip(0, 1)
    stats["TRANSITION_REPEAT_RATE"] = (
        (
            col_num(stats, "SOURCE_CHANGE_EVENTS", 0)
            - col_num(stats, "UNIQUE_TRANSITION_EDGE_COUNT", 0)
        ).clip(lower=0) / event_count
    ).clip(0, 1)
    stats["CHANGE_INTERVAL_IRREGULARITY"] = (
        col_num(stats, "STD_CHANGE_INTERVAL_DAYS", 0)
        / col_num(stats, "MEAN_CHANGE_INTERVAL_DAYS", 0).replace(0, np.nan)
    ).replace([np.inf, -np.inf], np.nan).fillna(0).clip(0, 10)
    return stats, events


def build_conflict_statistics(
    conflicts: pd.DataFrame,
    run_info: RunInfo,
) -> pd.DataFrame:
    keys = ["POPULATION", "ECGI", "FEATURE"]
    if conflicts is None or conflicts.empty:
        return pd.DataFrame(columns=keys)
    result = conflicts.groupby(keys, as_index=False).agg(
        SAME_DAY_CONFLICT_DAYS=("LOAD_DATE", "nunique"),
        SAME_DAY_CONFLICT_ROWS=("LOAD_DATE", "size"),
        MAX_SAME_DAY_UNIQUE_VALUES=("SOURCE_VALUES_UNIQUE_COUNT", "max"),
        LAST_SAME_DAY_CONFLICT_DATE=("LOAD_DATE", "max"),
    )
    result["LAST_SAME_DAY_CONFLICT_DATE"] = pd.to_datetime(
        result["LAST_SAME_DAY_CONFLICT_DATE"], errors="coerce"
    ).dt.normalize()
    result["CURRENT_DAY_SAME_DAY_CONFLICT_FLAG"] = result[
        "LAST_SAME_DAY_CONFLICT_DATE"
    ].eq(run_info.window_end).astype(int)
    result["RECURRENT_SAME_DAY_CONFLICT_FLAG"] = col_num(
        result, "SAME_DAY_CONFLICT_DAYS", 0
    ).ge(2).astype(int)
    return result


def feature_family(series: pd.Series) -> pd.Series:
    feature = series.fillna("").astype(str).str.upper()
    family = pd.Series("CATEGORICAL", index=feature.index, dtype=object)
    coordinate = feature.isin(["ANTENNALAT", "ANTENNALON"])
    family.loc[coordinate] = "COORDINATE"
    continuous = feature.map(
        lambda item: any(term in item for term in [
            "ALTITUDE", "AVERAGEALT", "AVGALT", "RADIUS", "RANGE",
            "ORIENTATION", "AZIMUTH", "BEARING", "UNCERT", "OPENING",
            "HEIGHT", "WIDTH",
        ])
    ) & ~coordinate
    family.loc[continuous] = "CONTINUOUS_NUMERIC"
    return family


def build_stability_context(
    data: pd.DataFrame,
    events: pd.DataFrame,
    run_info: RunInfo,
    args: argparse.Namespace,
) -> pd.DataFrame:
    """Build ECGI-level recency context without deciding anomaly eligibility.

    V9.7 incorrectly treated an adequately observed ECGI with no change in the
    trailing window as normal for the entire analysis window.  That whole-cell
    veto erased older high-frequency, reversal, rare, and multi-feature paths.
    V9.8 retains the same measurement as review context only.
    """
    candidates = data[data["POPULATION"].eq("CANDIDATE")].copy()
    if candidates.empty:
        return pd.DataFrame(columns=["ECGI"])
    # Coverage and latest-day presence must come from the same monitored
    # feature row.  Taking independent maxima can fabricate coherent coverage
    # from two different sparse features and falsely suppress the entire ECGI.
    candidates["COHERENT_RECENT_OBSERVED_DAYS"] = np.where(
        col_num(candidates, "LATEST_OBSERVED_ON_END_DATE_FLAG", 0).eq(1),
        col_num(candidates, "RECENT_OBSERVED_DAYS", 0),
        0,
    )
    observations = candidates.groupby("ECGI", as_index=False).agg(
        ECGI_RECENT_OBSERVED_DAYS=("COHERENT_RECENT_OBSERVED_DAYS", "max"),
        ECGI_LATEST_OBSERVED_ON_END_FLAG=("LATEST_OBSERVED_ON_END_DATE_FLAG", "max"),
    )
    candidate_events = events[events["POPULATION"].eq("CANDIDATE")].copy() if events is not None else pd.DataFrame()
    if candidate_events.empty:
        recent = pd.DataFrame(columns=["ECGI", "ECGI_RECENT_CHANGE_DAYS"])
        last = pd.DataFrame(columns=["ECGI", "ECGI_LAST_ANY_CHANGE_DATE"])
    else:
        recent = (
            candidate_events.loc[
                pd.to_datetime(candidate_events["LOAD_DATE"], errors="coerce").ge(run_info.stable_start)
            ]
            .groupby("ECGI", as_index=False)["LOAD_DATE"].nunique()
            .rename(columns={"LOAD_DATE": "ECGI_RECENT_CHANGE_DAYS"})
        )
        last = (
            candidate_events.groupby("ECGI", as_index=False)["LOAD_DATE"].max()
            .rename(columns={"LOAD_DATE": "ECGI_LAST_ANY_CHANGE_DATE"})
        )
    context = observations.merge(recent, on="ECGI", how="left").merge(last, on="ECGI", how="left")
    context["ECGI_RECENT_CHANGE_DAYS"] = col_num(context, "ECGI_RECENT_CHANGE_DAYS", 0)
    context["ECGI_LAST_ANY_CHANGE_DATE"] = pd.to_datetime(
        context["ECGI_LAST_ANY_CHANGE_DATE"], errors="coerce"
    ).dt.normalize()
    context["ECGI_DAYS_SINCE_LAST_ANY_CHANGE"] = (
        run_info.window_end - context["ECGI_LAST_ANY_CHANGE_DATE"]
    ).dt.days.fillna(9999).clip(lower=0)
    adequate = col_num(context, "ECGI_RECENT_OBSERVED_DAYS", 0).ge(
        int(args.stable_min_observed_days)
    ) & col_num(context, "ECGI_LATEST_OBSERVED_ON_END_FLAG", 0).eq(1)
    quiet = adequate & col_num(context, "ECGI_RECENT_CHANGE_DAYS", 0).eq(0)
    context["QUIET_LAST_N_DAYS_FLAG"] = quiet.astype(int)
    # Compatibility alias for existing consumers.  This field is descriptive
    # only and is structurally forbidden from suppressing a decision.
    context["STABLE_LAST_N_DAYS_FLAG"] = quiet.astype(int)
    context["STABILITY_STATUS"] = np.select(
        [quiet, adequate],
        [
            f"QUIET_LAST_{int(args.stable_days)}D_CONTEXT_ONLY",
            "RECENT_CHANGE_CONTEXT",
        ],
        default="RECENCY_CONTEXT_UNVERIFIED_INSUFFICIENT_RECENT_OBSERVATION",
    )
    return context


def build_feature_dataset(
    population: pd.DataFrame,
    metrics: pd.DataFrame,
    changes: pd.DataFrame,
    conflicts: pd.DataFrame,
    run_info: RunInfo,
    args: argparse.Namespace,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    header("PHASE 3 - SIMPLE TEMPORAL FEATURE DATASET AND TRAILING RECENCY CONTEXT")
    if metrics is None or metrics.empty:
        return pd.DataFrame(), add_reverse_transition_flags(changes)
    data = metrics.copy()
    change_stats, events = build_change_statistics(changes, run_info)
    conflict_stats = build_conflict_statistics(conflicts, run_info)
    keys = ["POPULATION", "ECGI", "FEATURE"]
    data = data.merge(change_stats, on=keys, how="left").merge(conflict_stats, on=keys, how="left")
    context_columns = [
        column for column in population.columns
        if column not in data.columns or column in ["POPULATION", "ECGI"]
    ]
    data = data.merge(
        population[context_columns].drop_duplicates(["POPULATION", "ECGI"]),
        on=["POPULATION", "ECGI"], how="left",
    )
    zero_columns = [
        "SOURCE_CHANGE_EVENTS", "SOURCE_CHANGE_DAYS", "BOUNDARY_CHANGE_EVENTS_RECOVERED",
        "A_B_A_REVERT_COUNT",
        "REVERSE_TRANSITION_COUNT", "UNIQUE_TRANSITION_EDGE_COUNT",
        "RECENT_CHANGE_DAYS", "RECENCY_WEIGHTED_ACTIVITY",
        "MEAN_CHANGE_INTERVAL_DAYS", "STD_CHANGE_INTERVAL_DAYS",
        "ABA_EVENT_RATE", "REVERSE_EVENT_RATE", "TRANSITION_REPEAT_RATE",
        "CHANGE_INTERVAL_IRREGULARITY", "SAME_DAY_CONFLICT_DAYS",
        "SAME_DAY_CONFLICT_ROWS", "MAX_SAME_DAY_UNIQUE_VALUES",
        "CURRENT_DAY_SAME_DAY_CONFLICT_FLAG", "RECURRENT_SAME_DAY_CONFLICT_FLAG",
    ]
    for column in zero_columns:
        data[column] = col_num(data, column, 0)
    data["DAYS_SINCE_LAST_CHANGE"] = col_num(data, "DAYS_SINCE_LAST_CHANGE", 9999)
    for column in [
        "LATEST_PREVIOUS_SOURCE_VALUE", "LATEST_CURRENT_SOURCE_VALUE",
        "LATEST_TRANSITION_EDGE",
    ]:
        data[column] = col_text(data, column, "")
    data["SITE_TYPE"] = col_text(data, "SITE_TYPE", "UNKNOWN")
    data["SITE_TYPE_MATCH_STATUS"] = col_text(data, "SITE_TYPE_MATCH_STATUS", "UNMATCHED")
    data["IS_COW_SITE"] = col_num(data, "IS_COW_SITE", 0).astype(int)
    data["FEATURE_FAMILY"] = feature_family(data["FEATURE"])
    data["MODE_IMPURITY_PERCENT"] = (
        100.0 - col_num(data, "MODE_SOURCE_VALUE_SHARE_PERCENT", 100)
    ).clip(0, 100)
    data["OBSERVED_COVERAGE_RATE"] = (
        col_num(data, "OBSERVED_DAYS", 0) / max(1, int(args.days))
    ).clip(0, 1)
    data["CHANGE_DAYS_RATE"] = (
        col_num(data, "SOURCE_CHANGE_DAYS", 0)
        / col_num(data, "OBSERVED_DAYS", 1).clip(lower=1)
    ).clip(0, 1)
    stability = build_stability_context(data, events, run_info, args)
    data = data.merge(stability, on="ECGI", how="left")
    for column, default in {
        "ECGI_RECENT_OBSERVED_DAYS": 0,
        "ECGI_LATEST_OBSERVED_ON_END_FLAG": 0,
        "ECGI_RECENT_CHANGE_DAYS": 0,
        "ECGI_DAYS_SINCE_LAST_ANY_CHANGE": 9999,
        "QUIET_LAST_N_DAYS_FLAG": 0,
        "STABLE_LAST_N_DAYS_FLAG": 0,
    }.items():
        data[column] = col_num(data, column, default)
    data["STABILITY_STATUS"] = col_text(
        data, "STABILITY_STATUS", "NOT_APPLICABLE_BASELINE"
    )
    if "ECGI_LAST_ANY_CHANGE_DATE" in data.columns:
        data["ECGI_LAST_ANY_CHANGE_DATE"] = pd.to_datetime(
            data["ECGI_LAST_ANY_CHANGE_DATE"], errors="coerce"
        ).dt.normalize()
    data["BASELINE_TRAINING_ELIGIBLE_FLAG"] = (
        data["POPULATION"].eq("BASELINE")
        & col_num(data, "OBSERVED_COVERAGE_RATE", 0).ge(float(args.min_baseline_coverage))
        & col_num(data, "LATEST_OBSERVED_ON_END_DATE_FLAG", 0).eq(1)
        & col_num(data, "SAME_DAY_CONFLICT_DAYS", 0).eq(0)
    ).astype(int)
    log(
        f"Candidate ECGIs quiet for trailing {int(args.stable_days)} days "
        f"(context only; suppressed=0): "
        f"{int(stability.get('QUIET_LAST_N_DAYS_FLAG', pd.Series(dtype=int)).sum()):,}",
        1,
    )
    candidate_rows = data["POPULATION"].eq("CANDIDATE")
    candidate_events = col_num(data.loc[candidate_rows], "SOURCE_CHANGE_DAYS", 0)
    log(
        "Candidate transition coverage: "
        f"changed feature rows={int(candidate_events.gt(0).sum()):,}; "
        f"one-date={int(candidate_events.eq(1).sum()):,}; "
        f"recurrent={int(candidate_events.ge(2).sum()):,}; "
        f"high-frequency-threshold-qualified={int(candidate_events.ge(int(args.high_frequency_min_change_days)).sum()):,}; "
        f"boundary events recovered={int(col_num(data.loc[candidate_rows], 'BOUNDARY_CHANGE_EVENTS_RECOVERED', 0).sum()):,}.",
        1,
    )
    return data, events


def prepare_model_matrix(frame: pd.DataFrame) -> pd.DataFrame:
    observed = col_num(frame, "OBSERVED_DAYS", 1).clip(lower=1)
    events = col_num(frame, "SOURCE_CHANGE_EVENTS", 0).clip(lower=0)
    matrix = pd.DataFrame(index=frame.index)
    matrix["CHANGE_DAYS_RATE"] = (
        col_num(frame, "SOURCE_CHANGE_DAYS", 0) / observed
    ).clip(0, 1)
    matrix["LOG_CHANGE_EVENTS"] = np.log1p(events)
    matrix["LOG_DISTINCT_VALUES"] = np.log1p(
        col_num(frame, "DISTINCT_SOURCE_VALUES", 0).clip(lower=0)
    )
    matrix["MODE_IMPURITY_FRACTION"] = (
        col_num(frame, "MODE_IMPURITY_PERCENT", 0) / 100.0
    ).clip(0, 1)
    matrix["ABA_EVENT_RATE"] = col_num(frame, "ABA_EVENT_RATE", 0).clip(0, 1)
    matrix["REVERSE_EVENT_RATE"] = col_num(frame, "REVERSE_EVENT_RATE", 0).clip(0, 1)
    matrix["RECENT_CHANGE_DAYS_RATE"] = (
        col_num(frame, "RECENT_CHANGE_DAYS", 0)
        / col_num(frame, "RECENT_OBSERVED_DAYS", 1).clip(lower=1)
    ).clip(0, 1)
    matrix["LOG_RECENCY_WEIGHTED_ACTIVITY"] = np.log1p(
        col_num(frame, "RECENCY_WEIGHTED_ACTIVITY", 0).clip(lower=0)
    )
    matrix["TRANSITION_REPEAT_RATE"] = col_num(
        frame, "TRANSITION_REPEAT_RATE", 0
    ).clip(0, 1)
    matrix["CHANGE_INTERVAL_IRREGULARITY"] = col_num(
        frame, "CHANGE_INTERVAL_IRREGULARITY", 0
    ).clip(0, 10)
    return matrix[MODEL_INPUT_COLUMNS].replace([np.inf, -np.inf], np.nan).fillna(0.0)


def stable_feature_seed(feature: object, base_seed: int) -> int:
    """Stable cross-process seed; Python's randomized hash is deliberately avoided."""
    digest = hashlib.sha256(str(feature).encode("utf-8")).digest()
    feature_part = int.from_bytes(digest[:4], byteorder="big", signed=False)
    return int((int(base_seed) + feature_part) % (2**31 - 1))


def isolation_seeds(args: argparse.Namespace) -> List[int]:
    seeds = [
        int(value.strip())
        for value in str(args.isolation_seeds).split(",")
        if value.strip()
    ]
    return list(dict.fromkeys(seeds)) or [int(args.random_state)]


def synthesize_temporal_anomalies(
    matrix: pd.DataFrame,
    random_state: int,
) -> pd.DataFrame:
    """Generate deterministic stress patterns used only for inner-fit tuning."""
    if matrix.empty:
        return matrix.copy()
    sample_count = min(len(matrix), 5000)
    synthetic = matrix.sample(
        sample_count, replace=False, random_state=int(random_state)
    ).reset_index(drop=True).copy()
    rng = np.random.default_rng(int(random_state))
    patterns = rng.integers(0, 4, size=len(synthetic))
    for position, pattern in enumerate(patterns):
        if pattern == 0:
            synthetic.loc[position, "CHANGE_DAYS_RATE"] = max(
                0.45, float(synthetic.loc[position, "CHANGE_DAYS_RATE"])
            )
            synthetic.loc[position, "LOG_CHANGE_EVENTS"] = max(
                math.log1p(10), float(synthetic.loc[position, "LOG_CHANGE_EVENTS"])
            )
            synthetic.loc[position, "RECENT_CHANGE_DAYS_RATE"] = max(
                0.55, float(synthetic.loc[position, "RECENT_CHANGE_DAYS_RATE"])
            )
        elif pattern == 1:
            synthetic.loc[position, "ABA_EVENT_RATE"] = 0.80
            synthetic.loc[position, "REVERSE_EVENT_RATE"] = 0.80
            synthetic.loc[position, "LOG_CHANGE_EVENTS"] = max(
                math.log1p(6), float(synthetic.loc[position, "LOG_CHANGE_EVENTS"])
            )
        elif pattern == 2:
            synthetic.loc[position, "MODE_IMPURITY_FRACTION"] = 0.75
            synthetic.loc[position, "LOG_DISTINCT_VALUES"] = max(
                math.log1p(5), float(synthetic.loc[position, "LOG_DISTINCT_VALUES"])
            )
            synthetic.loc[position, "TRANSITION_REPEAT_RATE"] = 0.80
        else:
            synthetic.loc[position, "CHANGE_INTERVAL_IRREGULARITY"] = 4.0
            synthetic.loc[position, "LOG_RECENCY_WEIGHTED_ACTIVITY"] = max(
                math.log1p(5),
                float(synthetic.loc[position, "LOG_RECENCY_WEIGHTED_ACTIVITY"]),
            )
            synthetic.loc[position, "RECENT_CHANGE_DAYS_RATE"] = max(
                0.65, float(synthetic.loc[position, "RECENT_CHANGE_DAYS_RATE"])
            )
    return synthetic[MODEL_INPUT_COLUMNS]


def choose_isolation_parameters(
    fit_pool: pd.DataFrame,
    args: argparse.Namespace,
    feature: str,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Tune strictly inside the outer fit pool; never inspect candidates/holdout."""
    default = {
        "n_estimators": int(args.isolation_n_estimators),
        "max_samples": int(args.isolation_max_samples),
        "max_features": float(args.isolation_max_features),
    }
    diagnostics: Dict[str, Any] = {
        "TUNING_REQUESTED": int(bool(args.tune_isolation_forest)),
        "TUNING_USED": 0,
        "TUNING_STATUS": "DISABLED",
        "TUNING_SYNTHETIC_AUC": np.nan,
        "TUNING_CANDIDATES_TESTED": 0,
    }
    if not args.tune_isolation_forest:
        return default, diagnostics
    if len(fit_pool) < int(args.isolation_tune_min_rows):
        diagnostics["TUNING_STATUS"] = "FALLBACK_INSUFFICIENT_INNER_ROWS"
        return default, diagnostics
    try:
        from sklearn.ensemble import IsolationForest
        from sklearn.metrics import roc_auc_score
    except Exception as exc:
        diagnostics["TUNING_STATUS"] = f"FALLBACK_IMPORT_ERROR:{type(exc).__name__}"
        return default, diagnostics

    tuning_seed = stable_feature_seed(feature, int(args.random_state) + 101)
    shuffled = fit_pool.sample(frac=1.0, random_state=tuning_seed).reset_index(drop=True)
    validation_count = max(2, int(round(0.25 * len(shuffled))))
    validation_count = min(validation_count, max(2, len(shuffled) // 2))
    inner_validation = shuffled.iloc[:validation_count]
    inner_train_pool = shuffled.iloc[validation_count:]
    if len(inner_validation) < 2 or len(inner_train_pool) < 2:
        diagnostics["TUNING_STATUS"] = "FALLBACK_INSUFFICIENT_INNER_SPLIT"
        return default, diagnostics
    inner_train = inner_train_pool.sample(
        min(len(inner_train_pool), int(args.isolation_tune_sample)),
        random_state=tuning_seed + 1,
    )
    synthetic = synthesize_temporal_anomalies(inner_validation, tuning_seed + 2)
    evaluation = pd.concat(
        [inner_validation.reset_index(drop=True), synthetic.reset_index(drop=True)],
        ignore_index=True,
    )
    labels = np.r_[np.zeros(len(inner_validation)), np.ones(len(synthetic))]
    design = [
        (300, 512, 0.70),
        (300, 2048, 1.00),
        (400, 2048, 0.80),
        (400, 4096, 1.00),
        (500, 512, 1.00),
        (500, 2048, 0.70),
        (500, 2048, 1.00),
        (500, 4096, 0.80),
    ]
    requested = (
        int(args.isolation_n_estimators),
        int(args.isolation_max_samples),
        float(args.isolation_max_features),
    )
    candidates = list(dict.fromkeys([requested, *design]))[
        : int(args.isolation_tune_max_configs)
    ]
    best_objective = -np.inf
    best_auc = np.nan
    selected = default.copy()
    tested = 0
    try:
        for n_estimators, max_samples, max_features in candidates:
            model = IsolationForest(
                n_estimators=int(n_estimators),
                max_samples=min(int(max_samples), len(inner_train)),
                max_features=float(max_features),
                contamination="auto",
                bootstrap=False,
                n_jobs=-1,
                random_state=tuning_seed + 3,
            )
            model.fit(inner_train)
            anomaly_scores = -model.score_samples(evaluation)
            auc = float(roc_auc_score(labels, anomaly_scores))
            effective_max_samples = min(int(max_samples), len(inner_train))
            complexity_penalty = (
                0.00001 * int(n_estimators) + 0.0000001 * effective_max_samples
            )
            objective = auc - complexity_penalty
            tested += 1
            if objective > best_objective:
                best_objective = objective
                best_auc = auc
                selected = {
                    "n_estimators": int(n_estimators),
                    "max_samples": int(max_samples),
                    "max_features": float(max_features),
                }
    except Exception as exc:
        diagnostics.update({
            "TUNING_STATUS": f"FALLBACK_TUNING_ERROR:{type(exc).__name__}",
            "TUNING_CANDIDATES_TESTED": tested,
        })
        return default, diagnostics
    if not np.isfinite(best_auc) or best_auc < float(args.isolation_tune_min_auc):
        diagnostics.update({
            "TUNING_STATUS": "FALLBACK_UNINFORMATIVE_SYNTHETIC_AUC",
            "TUNING_SYNTHETIC_AUC": best_auc,
            "TUNING_CANDIDATES_TESTED": tested,
        })
        return default, diagnostics
    diagnostics.update({
        "TUNING_USED": 1,
        "TUNING_STATUS": "TUNED_ON_INNER_FIT_POOL",
        "TUNING_SYNTHETIC_AUC": best_auc,
        "TUNING_CANDIDATES_TESTED": tested,
    })
    return selected, diagnostics


def add_feature_specific_isolation_forest(
    data: pd.DataFrame,
    args: argparse.Namespace,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    header("PHASE 4 - ROBUST POOLED IF ENSEMBLE WITH HELD-OUT NORMAL CALIBRATION")
    # Avoid a full multi-million-row sort/copy. Determinism is established per
    # feature below by sorting the baseline/candidate identity indices by ECGI.
    result = data.copy(deep=False)
    if not isinstance(result.index, pd.RangeIndex):
        result = result.reset_index(drop=True)
    result["MODEL_RAW_ANOMALY_SCORE"] = np.nan
    result["MODEL_NORMAL_PERCENTILE"] = 0.0
    result["MODEL_RISK_SCORE"] = 0.0
    result["MODEL_AVAILABLE_FLAG"] = 0
    result["MODEL_CALIBRATION_ROWS"] = 0
    result["MODEL_CONTEXT"] = "MODEL_UNAVAILABLE"
    result["MODEL_ENSEMBLE_SEED_COUNT"] = 0
    result["MODEL_PERCENTILE_IQR"] = 0.0
    result["MODEL_TAIL_VOTE_SHARE"] = 0.0
    result["MODEL_ONE_CHANGE_CONDITIONAL_PERCENTILE"] = 0.0
    result["MODEL_ONE_CHANGE_CONFORMAL_P_MAX"] = 1.0
    result["MODEL_ONE_CHANGE_REFERENCE_ROWS"] = 0
    result["MODEL_ONE_CHANGE_TAIL_VOTE_SHARE"] = 0.0
    result["MODEL_ONE_CHANGE_CONDITIONAL_AVAILABLE_FLAG"] = 0
    result["MODEL_ONE_CHANGE_AGE_BUCKET"] = one_change_age_bucket(
        col_num(result, "DAYS_SINCE_LAST_CHANGE", 9999)
    )
    result["MODEL_SELECTED_N_ESTIMATORS"] = 0
    result["MODEL_SELECTED_MAX_SAMPLES"] = 0
    result["MODEL_SELECTED_MAX_FEATURES"] = 0.0
    result["MODEL_TUNING_USED_FLAG"] = 0
    if result.empty or args.disable_isolation_forest:
        log("Isolation Forest disabled or dataset empty; RARE_CHANGES model branch unavailable.", 1)
        return result, pd.DataFrame()
    try:
        from sklearn.ensemble import IsolationForest
    except Exception as exc:
        log(f"scikit-learn unavailable; RARE_CHANGES model branch disabled ({exc}).", 1)
        return result, pd.DataFrame()

    records: List[Dict[str, Any]] = []
    for feature in sorted(result["FEATURE"].dropna().astype(str).unique()):
        feature_mask = result["FEATURE"].eq(feature)
        baseline_index = pd.Index(
            result.loc[
                feature_mask & result["BASELINE_TRAINING_ELIGIBLE_FLAG"].eq(1),
                ["ECGI"],
            ].sort_values("ECGI", kind="mergesort").index
        )
        total_candidate_index = result.index[
            feature_mask & result["POPULATION"].eq("CANDIDATE")
        ]
        # A model score can affect a decision only when a real inter-day source
        # transition exists. Current invalid values and same-day conflicts have
        # their own objective rule branches, so scoring millions of unchanged
        # candidate rows adds cost without changing any incident decision.
        candidate_index = pd.Index(
            result.loc[
                feature_mask & result["POPULATION"].eq("CANDIDATE")
                & col_num(result, "SOURCE_CHANGE_EVENTS", 0).gt(0),
                ["ECGI"],
            ].sort_values("ECGI", kind="mergesort").index
        )
        if len(candidate_index) == 0:
            continue
        if len(baseline_index) < int(args.min_feature_model_rows):
            records.append({
                "FEATURE": feature,
                "STATUS": "SKIPPED_INSUFFICIENT_REAL_NORMAL_HISTORY",
                "NORMAL_ROWS": len(baseline_index),
                "CANDIDATE_ROWS": len(candidate_index),
                "CANDIDATE_FEATURE_ROWS_TOTAL": len(total_candidate_index),
            })
            continue
        split_seed = stable_feature_seed(feature, int(args.random_state))
        shuffled = pd.Series(baseline_index).sample(
            frac=1.0, random_state=split_seed
        ).to_numpy()
        calibration_count = max(
            int(args.min_calibration_rows),
            int(round(len(shuffled) * float(args.calibration_fraction))),
        )
        calibration_count = min(calibration_count, max(1, len(shuffled) // 2))
        calibration_index = pd.Index(shuffled[:calibration_count])
        fit_index = pd.Index(shuffled[calibration_count:])
        # Honor --min-feature-model-rows for small validation runs while the
        # production default (500) still requires a substantial fit partition.
        if len(fit_index) < max(2, int(args.min_feature_model_rows) // 2):
            records.append({
                "FEATURE": feature,
                "STATUS": "SKIPPED_INSUFFICIENT_FIT_PARTITION",
                "NORMAL_ROWS": len(baseline_index),
                "FIT_ROWS": len(fit_index),
                "CALIBRATION_ROWS": len(calibration_index),
                "CANDIDATE_ROWS": len(candidate_index),
                "CANDIDATE_FEATURE_ROWS_TOTAL": len(total_candidate_index),
            })
            continue
        # Materialize the numeric matrix one feature at a time. At production
        # scale this avoids a multi-million-row global pandas matrix while
        # preserving the exact fit/calibration population and scores.
        needed_index = pd.Index(baseline_index).union(
            pd.Index(candidate_index), sort=False
        )
        feature_matrix = prepare_model_matrix(result.loc[needed_index])
        fit_pool = feature_matrix.loc[fit_index]
        calibration_matrix = feature_matrix.loc[calibration_index]
        candidate_matrix = feature_matrix.loc[candidate_index]
        one_change_calibration_base = (
            col_num(result.loc[calibration_index], "SOURCE_CHANGE_DAYS", 0).eq(1)
            & col_num(result.loc[calibration_index], "SOURCE_CHANGE_EVENTS", 0).eq(1)
            & col_num(
                result.loc[calibration_index], "OBSERVED_COVERAGE_RATE", 0
            ).ge(float(args.one_change_min_coverage))
        )
        calibration_age_bucket = result.loc[
            calibration_index, "MODEL_ONE_CHANGE_AGE_BUCKET"
        ].astype(str)
        candidate_age_bucket = result.loc[
            candidate_index, "MODEL_ONE_CHANGE_AGE_BUCKET"
        ].astype(str)
        candidate_one_change_base = (
            col_num(result.loc[candidate_index], "SOURCE_CHANGE_DAYS", 0).eq(1)
            & col_num(result.loc[candidate_index], "SOURCE_CHANGE_EVENTS", 0).eq(1)
            & col_num(
                result.loc[candidate_index], "OBSERVED_COVERAGE_RATE", 0
            ).ge(float(args.one_change_min_coverage))
            & col_num(
                result.loc[candidate_index],
                "LATEST_OBSERVED_ON_END_DATE_FLAG",
                0,
            ).eq(1)
        )
        conditional_bucket_counts = {
            bucket: int((one_change_calibration_base & calibration_age_bucket.eq(bucket)).sum())
            for bucket in ["AGE_0_3D", "AGE_4_7D", "AGE_8_14D", "AGE_15_30D"]
        }
        candidate_reference_rows = np.where(
            candidate_one_change_base.to_numpy(),
            candidate_age_bucket.map(conditional_bucket_counts)
            .fillna(0).astype(int).to_numpy(),
            0,
        )
        candidate_reference_available = (
            candidate_reference_rows >= int(args.min_one_change_calibration_rows)
        ) & candidate_one_change_base.to_numpy() & candidate_age_bucket.ne(
            "AGE_UNAVAILABLE"
        ).to_numpy()
        selected, tuning = choose_isolation_parameters(fit_pool, args, feature)
        percentile_members: List[np.ndarray] = []
        one_change_percentile_members: List[np.ndarray] = []
        one_change_p_members: List[np.ndarray] = []
        raw_members: List[np.ndarray] = []
        calibration_tail_rates: List[float] = []
        seeds = isolation_seeds(args)
        fit_rows_per_seed: List[int] = []
        for seed in seeds:
            model_seed = stable_feature_seed(feature, seed)
            fit_matrix = fit_pool.sample(
                min(len(fit_pool), int(args.isolation_fit_sample)),
                random_state=model_seed,
            )
            fit_rows_per_seed.append(len(fit_matrix))
            model = IsolationForest(
                n_estimators=int(selected["n_estimators"]),
                max_samples=min(int(selected["max_samples"]), len(fit_matrix)),
                max_features=float(selected["max_features"]),
                contamination="auto",
                bootstrap=False,
                n_jobs=-1,
                random_state=model_seed,
            )
            model.fit(fit_matrix)
            calibration_scores = -model.score_samples(calibration_matrix)
            candidate_scores = -model.score_samples(candidate_matrix)
            percentiles = empirical_percentile(calibration_scores, candidate_scores)
            percentile_members.append(percentiles)
            conditional_percentiles = np.zeros(len(candidate_index), dtype=float)
            conditional_p = np.ones(len(candidate_index), dtype=float)
            for bucket, reference_rows in conditional_bucket_counts.items():
                if reference_rows < int(args.min_one_change_calibration_rows):
                    continue
                reference_mask = (
                    one_change_calibration_base & calibration_age_bucket.eq(bucket)
                ).to_numpy()
                target_mask = (
                    candidate_one_change_base & candidate_age_bucket.eq(bucket)
                ).to_numpy()
                if not target_mask.any():
                    continue
                reference_scores = calibration_scores[reference_mask]
                conditional_percentiles[target_mask] = empirical_percentile(
                    reference_scores, candidate_scores[target_mask]
                )
                conditional_p[target_mask] = conformal_upper_tail_p(
                    reference_scores, candidate_scores[target_mask]
                )
            one_change_percentile_members.append(conditional_percentiles)
            one_change_p_members.append(conditional_p)
            raw_members.append(candidate_scores)
            calibration_percentiles = empirical_percentile(
                calibration_scores, calibration_scores
            )
            calibration_tail_rates.append(float(np.mean(
                calibration_percentiles >= float(args.rare_model_percentile)
            )))
        percentile_stack = np.vstack(percentile_members)
        raw_stack = np.vstack(raw_members)
        median_percentile = np.median(percentile_stack, axis=0)
        mean_raw_score = np.mean(raw_stack, axis=0)
        percentile_iqr = np.percentile(percentile_stack, 75, axis=0) - np.percentile(
            percentile_stack, 25, axis=0
        )
        tail_vote_share = np.mean(
            percentile_stack >= float(args.rare_model_percentile), axis=0
        )
        one_change_stack = np.vstack(one_change_percentile_members)
        one_change_p_stack = np.vstack(one_change_p_members)
        one_change_median_percentile = np.median(one_change_stack, axis=0)
        one_change_p_max = np.max(one_change_p_stack, axis=0)
        one_change_tail_vote_share = np.mean(
            one_change_p_stack <= float(args.one_change_model_alpha), axis=0
        )
        result.loc[candidate_index, "MODEL_RAW_ANOMALY_SCORE"] = mean_raw_score
        result.loc[candidate_index, "MODEL_NORMAL_PERCENTILE"] = median_percentile
        baseline = float(args.model_risk_baseline_percentile)
        result.loc[candidate_index, "MODEL_RISK_SCORE"] = np.clip(
            100.0 * (median_percentile - baseline) / max(1e-9, 100.0 - baseline), 0, 100
        )
        result.loc[candidate_index, "MODEL_AVAILABLE_FLAG"] = 1
        result.loc[candidate_index, "MODEL_CALIBRATION_ROWS"] = len(calibration_index)
        result.loc[candidate_index, "MODEL_CONTEXT"] = (
            f"{feature}|ALL_SITE_TYPES_POOLED_CONTEXT_ONLY"
        )
        result.loc[candidate_index, "MODEL_ENSEMBLE_SEED_COUNT"] = len(seeds)
        result.loc[candidate_index, "MODEL_PERCENTILE_IQR"] = percentile_iqr
        result.loc[candidate_index, "MODEL_TAIL_VOTE_SHARE"] = tail_vote_share
        result.loc[
            candidate_index, "MODEL_ONE_CHANGE_CONDITIONAL_PERCENTILE"
        ] = one_change_median_percentile
        result.loc[
            candidate_index, "MODEL_ONE_CHANGE_CONFORMAL_P_MAX"
        ] = one_change_p_max
        result.loc[
            candidate_index, "MODEL_ONE_CHANGE_REFERENCE_ROWS"
        ] = candidate_reference_rows
        result.loc[
            candidate_index, "MODEL_ONE_CHANGE_TAIL_VOTE_SHARE"
        ] = one_change_tail_vote_share
        result.loc[
            candidate_index, "MODEL_ONE_CHANGE_CONDITIONAL_AVAILABLE_FLAG"
        ] = candidate_reference_available.astype(int)
        result.loc[candidate_index, "MODEL_SELECTED_N_ESTIMATORS"] = int(
            selected["n_estimators"]
        )
        result.loc[candidate_index, "MODEL_SELECTED_MAX_SAMPLES"] = int(
            selected["max_samples"]
        )
        result.loc[candidate_index, "MODEL_SELECTED_MAX_FEATURES"] = float(
            selected["max_features"]
        )
        result.loc[candidate_index, "MODEL_TUNING_USED_FLAG"] = int(
            tuning["TUNING_USED"]
        )
        records.append({
            "FEATURE": feature,
            "STATUS": "FITTED",
            "NORMAL_ROWS": len(baseline_index),
            "FIT_POOL_ROWS": len(fit_pool),
            "FIT_ROWS_PER_SEED_MIN": min(fit_rows_per_seed),
            "FIT_ROWS_PER_SEED_MAX": max(fit_rows_per_seed),
            "CALIBRATION_ROWS": len(calibration_index),
            "ONE_CHANGE_CALIBRATION_ROWS": int(sum(conditional_bucket_counts.values())),
            "ONE_CHANGE_CALIBRATION_BUCKET_COUNTS": json.dumps(
                conditional_bucket_counts, sort_keys=True
            ),
            "ONE_CHANGE_CALIBRATION_STATUS": (
                "AVAILABLE_IN_AT_LEAST_ONE_AGE_BUCKET"
                if max(conditional_bucket_counts.values(), default=0)
                >= int(args.min_one_change_calibration_rows)
                else "INSUFFICIENT_CONDITIONAL_REFERENCE"
            ),
            "CANDIDATE_ROWS": len(candidate_index),
            "CANDIDATE_FEATURE_ROWS_TOTAL": len(total_candidate_index),
            "CANDIDATE_CHANGED_ROWS_SCORED": len(candidate_index),
            "N_ESTIMATORS": int(selected["n_estimators"]),
            "MAX_SAMPLES": int(selected["max_samples"]),
            "MAX_FEATURES": float(selected["max_features"]),
            "ENSEMBLE_SEEDS": ",".join(map(str, seeds)),
            "ENSEMBLE_SEED_COUNT": len(seeds),
            "CALIBRATION_TAIL_RATE_MEAN": float(np.mean(calibration_tail_rates)),
            "MODEL_CONTEXT": f"{feature}|ALL_SITE_TYPES_POOLED_CONTEXT_ONLY",
            **tuning,
        })
    calibration = pd.DataFrame(records)
    scored = int(
        result.loc[result["POPULATION"].eq("CANDIDATE"), "MODEL_AVAILABLE_FLAG"].sum()
    )
    log(
        f"Ensemble-scored candidate feature rows={scored:,}; "
        f"seeds={','.join(map(str, isolation_seeds(args)))}; "
        f"fine_tune={'enabled' if args.tune_isolation_forest else 'disabled'}.",
        1,
    )
    return result, calibration


def assign_feature_clusters(data: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    """One compact feature decision tree; no legacy cluster aliases or score gate."""
    header("PHASE 5A - THREE FEATURE-LEVEL PATTERNS")
    result = data.copy()
    candidate = result["POPULATION"].eq("CANDIDATE")
    quiet_column = (
        "QUIET_LAST_N_DAYS_FLAG"
        if "QUIET_LAST_N_DAYS_FLAG" in result.columns
        else "STABLE_LAST_N_DAYS_FLAG"
    )
    quiet = col_num(result, quiet_column, 0).eq(1)
    # V9.7 used ``candidate & ~quiet`` here and thereby discarded every
    # qualifying full-window pattern whose last event was outside the trailing
    # recency window. Recency is context/ranking only in V9.8.
    active = candidate
    change_days = col_num(result, "SOURCE_CHANGE_DAYS", 0)
    change_events = col_num(result, "SOURCE_CHANGE_EVENTS", 0)
    reversals = (
        col_num(result, "A_B_A_REVERT_COUNT", 0)
        + col_num(result, "REVERSE_TRANSITION_COUNT", 0)
    )
    recurrent = change_days.ge(2) & change_events.ge(2)
    model_available = col_num(result, "MODEL_AVAILABLE_FLAG", 0).eq(1)
    model_percentile = col_num(result, "MODEL_NORMAL_PERCENTILE", 0)
    one_change_conditional_percentile = col_num(
        result, "MODEL_ONE_CHANGE_CONDITIONAL_PERCENTILE", 0
    )
    one_change_model_evidence = (
        candidate
        & change_days.eq(1)
        & change_events.eq(1)
        & col_num(result, "MODEL_ONE_CHANGE_CONDITIONAL_AVAILABLE_FLAG", 0).eq(1)
        & col_num(result, "MODEL_ONE_CHANGE_CONFORMAL_P_MAX", 1.0).le(
            float(args.one_change_model_alpha)
        )
        & col_num(result, "OBSERVED_COVERAGE_RATE", 0).ge(
            float(args.one_change_min_coverage)
        )
        & col_num(result, "LATEST_OBSERVED_ON_END_DATE_FLAG", 0).eq(1)
    )
    conflict_evidence = (
        col_num(result, "CURRENT_DAY_SAME_DAY_CONFLICT_FLAG", 0).eq(1)
        | col_num(result, "RECURRENT_SAME_DAY_CONFLICT_FLAG", 0).eq(1)
    )

    # Preserve raw evidence flags for audit.  The final feature label is exclusive.
    raw_backforth = candidate & recurrent & reversals.gt(0)
    raw_high_frequency = candidate & change_days.ge(int(args.high_frequency_min_change_days))
    raw_rare_model = (
        candidate & recurrent & model_available
        & model_percentile.ge(float(args.rare_model_percentile))
    )
    raw_rare_conflict = candidate & conflict_evidence
    raw_rare = raw_rare_model | raw_rare_conflict | one_change_model_evidence

    backforth = active & raw_backforth
    high_frequency = active & ~backforth & raw_high_frequency
    rare = active & ~backforth & ~high_frequency & raw_rare

    result["RAW_REAL_BACK_AND_FORTH_FLAG"] = raw_backforth.astype(int)
    result["RAW_HIGH_FREQUENCY_CHANGES_FLAG"] = raw_high_frequency.astype(int)
    result["RAW_BACKFORTH_AND_HIGH_FREQUENCY_OVERLAP_FLAG"] = (
        raw_backforth & raw_high_frequency
    ).astype(int)
    result["RAW_RARE_RECURRENT_MODEL_FLAG"] = raw_rare_model.astype(int)
    result["RAW_RARE_CONFLICT_FLAG"] = raw_rare_conflict.astype(int)
    result["RAW_RARE_ONE_CHANGE_CONDITIONAL_MODEL_FLAG"] = (
        one_change_model_evidence.astype(int)
    )
    result["RAW_QUALIFIED_FEATURE_FLAG"] = (
        raw_backforth | raw_high_frequency | raw_rare
    ).astype(int)
    result["RECURRENT_CHANGE_FLAG"] = (candidate & recurrent).astype(int)
    result["REAL_BACK_AND_FORTH_FLAG"] = backforth.astype(int)
    result["HIGH_FREQUENCY_CHANGES_FLAG"] = high_frequency.astype(int)
    result["RARE_CHANGES_FLAG"] = rare.astype(int)
    result["ONE_DAY_RARE_GUARD_FLAG"] = (
        candidate & change_days.le(1)
        & model_percentile.ge(float(args.rare_model_percentile))
        & ~conflict_evidence
        & ~one_change_model_evidence
    ).astype(int)
    # Compatibility field now records the absence of an override.  The
    # counterfactual field quantifies what the defective V9.7 policy would have
    # removed, making recall loss visible in every production run.
    result["STABLE_CELL_OVERRIDE_FLAG"] = 0
    result["WOULD_OLD_STABILITY_OVERRIDE_DROP_FEATURE_FLAG"] = (
        quiet & (backforth | high_frequency | rare)
    ).astype(int)
    result["FEATURE_FINAL_CLUSTER"] = np.select(
        [backforth, high_frequency, rare],
        ["REAL_BACK_AND_FORTH", "HIGH_FREQUENCY_CHANGES", "RARE_CHANGES"],
        default="",
    )

    unified_reversal_count = np.maximum(
        col_num(result, "A_B_A_REVERT_COUNT", 0),
        col_num(result, "REVERSE_TRANSITION_COUNT", 0),
    )
    back_score = np.minimum(
        99.0,
        72.0 + 8.0 * unified_reversal_count
        + 2.0 * np.maximum(change_days - 2, 0),
    )
    high_score = np.minimum(
        99.0,
        70.0
        + 4.0 * np.maximum(change_days - int(args.high_frequency_min_change_days), 0)
        + 2.0 * col_num(result, "RECENT_CHANGE_DAYS", 0)
        + np.where(model_percentile.ge(float(args.model_support_percentile)), 3.0, 0.0),
    )
    rare_tail = (
        (model_percentile - float(args.rare_model_percentile))
        / max(1e-9, 100.0 - float(args.rare_model_percentile))
    ).clip(0, 1)
    rare_model_score = np.minimum(
        99.0,
        62.0 + 26.0 * rare_tail + 2.0 * np.minimum(np.maximum(change_days - 2, 0), 4),
    )
    rare_conflict_score = np.minimum(
        99.0,
        78.0
        + 6.0 * col_num(result, "RECURRENT_SAME_DAY_CONFLICT_FLAG", 0)
        + 2.0 * np.maximum(col_num(result, "SAME_DAY_CONFLICT_DAYS", 0) - 2, 0),
    )
    rare_score = np.where(
        raw_rare_conflict,
        np.maximum(rare_model_score, rare_conflict_score),
        rare_model_score,
    )
    one_change_tail = (
        (float(args.one_change_model_alpha)
         - col_num(result, "MODEL_ONE_CHANGE_CONFORMAL_P_MAX", 1.0))
        / max(1e-12, float(args.one_change_model_alpha))
    ).clip(0, 1)
    one_change_model_score = np.minimum(99.0, 72.0 + 24.0 * one_change_tail)
    rare_score = np.where(
        one_change_model_evidence,
        np.maximum(rare_score, one_change_model_score),
        rare_score,
    )
    result["FEATURE_CLUSTER_SCORE"] = np.select(
        [backforth, high_frequency, rare],
        [back_score, high_score, rare_score],
        default=0.0,
    )
    result["FEATURE_REPORTABLE_FLAG"] = result["FEATURE_FINAL_CLUSTER"].ne("").astype(int)
    result["DETECTION_METHOD"] = np.select(
        [backforth, high_frequency & model_percentile.ge(float(args.model_support_percentile)),
         high_frequency, rare & raw_rare_conflict & raw_rare_model,
         rare & raw_rare_conflict, rare & one_change_model_evidence, rare],
        ["SOURCE_SEQUENCE_RULE", "CHANGE_FREQUENCY_RULE_WITH_MODEL_SUPPORT",
         "CHANGE_FREQUENCY_RULE", "SOURCE_CONFLICT_WITH_MODEL_SUPPORT",
         "CURRENT_OR_RECURRENT_SOURCE_CONFLICT",
         "ONE_CHANGE_CONDITIONAL_NORMAL_CALIBRATED_ISOLATION_FOREST",
         "NORMAL_CALIBRATED_ISOLATION_FOREST"],
        default="NONE",
    )
    result["COW_REVIEW_NOTE"] = np.where(
        col_num(result, "IS_COW_SITE", 0).eq(1),
        "COW SITE: review mobility as operational context; SITE_TYPE did not alter detection or score.",
        "",
    )
    result["FEATURE_DECISION_REASON"] = np.select(
        [backforth, high_frequency, rare & raw_rare_conflict,
         rare & one_change_model_evidence, rare,
         candidate & change_days.eq(1), candidate & recurrent & ~raw_rare,
         candidate & quiet],
        [
            "Actual reverse transition / A->B->A source path",
            f"At least {int(args.high_frequency_min_change_days)} distinct change dates on one feature",
            "Current-day or recurrent same-day source values conflict for this feature",
            f"One change date is extreme versus held-out same-age normal one-change histories (worst-seed conformal p <= {float(args.one_change_model_alpha):.4f})",
            f"At least two change dates and model percentile >= {float(args.rare_model_percentile):.1f} versus held-out real normal history",
            "One change date only; cannot be rare or high frequency",
            "Recurrent but not back-and-forth, high frequency, or unusual enough versus normal history",
            f"No source change in the trailing {int(args.stable_days)} days; recency context only, with no qualifying full-window pattern",
        ],
        default="No monitored source-history anomaly",
    )
    log(
        "Feature clusters: "
        + str(result.loc[result["FEATURE_REPORTABLE_FLAG"].eq(1), "FEATURE_FINAL_CLUSTER"].value_counts().to_dict()),
        1,
    )
    return result


def build_multi_feature_topology(
    data: pd.DataFrame,
    events: pd.DataFrame,
    args: argparse.Namespace,
) -> pd.DataFrame:
    """One anti-bulk topology rule: features must genuinely spread across dates."""
    header("PHASE 5B - MULTI-FEATURE DATE TOPOLOGY")
    result = data.copy()
    defaults: Dict[str, Any] = {
        "MULTI_CHANGED_FEATURE_COUNT": 0,
        "MULTI_UNIQUE_CHANGE_DATES": 0,
        "MULTI_FEATURE_DATE_EVENTS": 0,
        "MULTI_MAX_FEATURES_ON_ONE_DATE": 0,
        "MULTI_MAX_SAME_DATE_FEATURE_SHARE": 0.0,
        "MULTI_DISTINCT_FEATURE_DATE_PATTERNS": 0,
        "RAW_MULTI_FEATURE_TOPOLOGY_FLAG": 0,
        "MULTI_FEATURE_CHANGES_FLAG": 0,
        "MULTI_FEATURE_SCORE": 0.0,
    }
    for name, default in defaults.items():
        result[name] = default
    result["MULTI_FEATURE_CONTRIBUTOR_FLAG"] = 0
    if events is None or events.empty:
        return result
    # All candidate histories participate.  V9.7 removed quiet ECGIs before
    # topology construction, so older asynchronous multi-feature incidents
    # could never qualify.
    active_ecgis = set(
        result.loc[result["POPULATION"].eq("CANDIDATE"), "ECGI"].astype(str)
    )
    event_data = events[
        events["POPULATION"].eq("CANDIDATE")
        & events["ECGI"].astype(str).isin(active_ecgis)
    ][["ECGI", "FEATURE", "LOAD_DATE"]].copy()
    event_data["LOAD_DATE"] = pd.to_datetime(event_data["LOAD_DATE"], errors="coerce").dt.normalize()
    event_data = event_data.dropna(subset=["LOAD_DATE"]).drop_duplicates()
    if event_data.empty:
        return result
    per_feature = event_data.groupby(["ECGI", "FEATURE"], as_index=False).agg(
        FEATURE_CHANGE_DAYS=("LOAD_DATE", "nunique"),
        FEATURE_DATE_PATTERN=(
            "LOAD_DATE",
            lambda values: "|".join(
                pd.to_datetime(values, errors="coerce").dropna().drop_duplicates()
                .sort_values().dt.strftime("%Y-%m-%d").tolist()
            ),
        ),
    )
    feature_summary = per_feature.groupby("ECGI", as_index=False).agg(
        MULTI_CHANGED_FEATURE_COUNT=("FEATURE", "nunique"),
        MULTI_DISTINCT_FEATURE_DATE_PATTERNS=("FEATURE_DATE_PATTERN", "nunique"),
    )
    per_day = event_data.groupby(["ECGI", "LOAD_DATE"], as_index=False).agg(
        FEATURES_CHANGED_ON_DATE=("FEATURE", "nunique")
    )
    date_summary = per_day.groupby("ECGI", as_index=False).agg(
        MULTI_UNIQUE_CHANGE_DATES=("LOAD_DATE", "nunique"),
        MULTI_FEATURE_DATE_EVENTS=("FEATURES_CHANGED_ON_DATE", "sum"),
        MULTI_MAX_FEATURES_ON_ONE_DATE=("FEATURES_CHANGED_ON_DATE", "max"),
    )
    topology = feature_summary.merge(date_summary, on="ECGI", how="inner")
    topology["MULTI_MAX_SAME_DATE_FEATURE_SHARE"] = (
        col_num(topology, "MULTI_MAX_FEATURES_ON_ONE_DATE", 0)
        / col_num(topology, "MULTI_FEATURE_DATE_EVENTS", 1).clip(lower=1)
    )
    raw_topology = (
        col_num(topology, "MULTI_CHANGED_FEATURE_COUNT", 0).ge(
            int(args.multi_feature_min_features)
        )
        & col_num(topology, "MULTI_UNIQUE_CHANGE_DATES", 0).ge(
            int(args.multi_feature_min_dates)
        )
        & col_num(topology, "MULTI_DISTINCT_FEATURE_DATE_PATTERNS", 0).ge(2)
    )
    multi = raw_topology & col_num(
        topology, "MULTI_MAX_SAME_DATE_FEATURE_SHARE", 1
    ).le(float(args.multi_max_dominant_date_share))
    topology["RAW_MULTI_FEATURE_TOPOLOGY_FLAG"] = raw_topology.astype(int)
    topology["MULTI_FEATURE_CHANGES_FLAG"] = multi.astype(int)
    topology["MULTI_FEATURE_SCORE"] = np.where(
        multi,
        np.minimum(
            99.0,
            68.0
            + 5.0 * np.maximum(
                col_num(topology, "MULTI_CHANGED_FEATURE_COUNT", 0)
                - int(args.multi_feature_min_features), 0
            )
            + 3.0 * np.maximum(
                col_num(topology, "MULTI_UNIQUE_CHANGE_DATES", 0)
                - int(args.multi_feature_min_dates), 0
            )
            + 2.0 * np.maximum(
                col_num(topology, "MULTI_DISTINCT_FEATURE_DATE_PATTERNS", 0) - 2, 0
            ),
        ),
        0.0,
    )
    result = result.drop(columns=list(defaults), errors="ignore").merge(
        topology, on="ECGI", how="left", validate="many_to_one"
    )
    for name, default in defaults.items():
        result[name] = col_num(result, name, float(default))
    changed_feature_keys = set(
        zip(per_feature["ECGI"].astype(str), per_feature["FEATURE"].astype(str))
    )
    is_changed_feature = pd.Series(
        [
            (str(ecgi), str(feature)) in changed_feature_keys
            for ecgi, feature in zip(result["ECGI"], result["FEATURE"])
        ],
        index=result.index,
    )
    result["MULTI_FEATURE_CONTRIBUTOR_FLAG"] = (
        result["POPULATION"].eq("CANDIDATE")
        & col_num(result, "MULTI_FEATURE_CHANGES_FLAG", 0).eq(1)
        & is_changed_feature
    ).astype(int)
    log(
        f"Raw >={int(args.multi_feature_min_features)}-feature/multi-date topology ECGIs={int(topology['RAW_MULTI_FEATURE_TOPOLOGY_FLAG'].sum()):,}; "
        f"final non-bulk multi-feature ECGIs={int(topology['MULTI_FEATURE_CHANGES_FLAG'].sum()):,}",
        1,
    )
    return result


def _finalize_signature_dominant(
    data: pd.DataFrame,
    events: pd.DataFrame,
    args: argparse.Namespace,
) -> pd.DataFrame:
    result = assign_feature_clusters(data, args)
    result = build_multi_feature_topology(result, events, args)
    result["REPORTABLE_EVIDENCE_ROW_FLAG"] = (
        col_num(result, "FEATURE_REPORTABLE_FLAG", 0).eq(1)
        | col_num(result, "MULTI_FEATURE_CONTRIBUTOR_FLAG", 0).eq(1)
    ).astype(int)
    result["EVIDENCE_ROLE"] = np.select(
        [
            col_num(result, "FEATURE_REPORTABLE_FLAG", 0).eq(1)
            & col_num(result, "MULTI_FEATURE_CONTRIBUTOR_FLAG", 0).eq(1),
            col_num(result, "FEATURE_REPORTABLE_FLAG", 0).eq(1),
            col_num(result, "MULTI_FEATURE_CONTRIBUTOR_FLAG", 0).eq(1),
        ],
        ["BOTH", "INDIVIDUAL_FEATURE_PATTERN", "MULTI_FEATURE_TOPOLOGY_CONTRIBUTOR"],
        default="NONE",
    )
    quiet_column = (
        "QUIET_LAST_N_DAYS_FLAG"
        if "QUIET_LAST_N_DAYS_FLAG" in result.columns
        else "STABLE_LAST_N_DAYS_FLAG"
    )
    quiet = col_num(result, quiet_column, 0).eq(1)
    reportable = col_num(result, "REPORTABLE_EVIDENCE_ROW_FLAG", 0).eq(1)
    result["WOULD_OLD_STABILITY_OVERRIDE_DROP_FLAG"] = (
        result["POPULATION"].eq("CANDIDATE") & quiet & reportable
    ).astype(int)
    # Explicit invariant/audit field: no qualifying row may be deleted merely
    # because it is quiet in the trailing recency window.
    result["DROPPED_BY_RECENCY_FLAG"] = 0
    # Hard invariants requested by the user.
    if result["FEATURE"].eq("PHYSICALCELLID").any():
        raise AssertionError("PHYSICALCELLID leaked into the V9.8 engineered feature dataset.")
    one_day_rare = (
        col_num(result, "SOURCE_CHANGE_DAYS", 0).le(1)
        & col_num(result, "RARE_CHANGES_FLAG", 0).eq(1)
        & col_num(
            result, "RAW_RARE_ONE_CHANGE_CONDITIONAL_MODEL_FLAG", 0
        ).eq(0)
        & col_num(result, "RAW_RARE_CONFLICT_FLAG", 0).eq(0)
    )
    if one_day_rare.any():
        raise AssertionError(
            "A one-change-day row reached RARE_CHANGES without conditional-normal "
            "model discovery or objective source-conflict evidence."
        )
    if col_num(result, "DROPPED_BY_RECENCY_FLAG", 0).sum() != 0:
        raise AssertionError("Recency context deleted otherwise-qualified evidence.")
    return result


def _copy_mode1_decisions(data: pd.DataFrame) -> pd.DataFrame:
    """Preserve every V9.8.4 decision field before constructing Mode 2."""
    result = data.copy()
    if result.empty:
        return result
    decision_columns = [
        "FEATURE_REPORTABLE_FLAG", "FEATURE_FINAL_CLUSTER",
        "FEATURE_CLUSTER_SCORE", "REPORTABLE_EVIDENCE_ROW_FLAG",
        "EVIDENCE_ROLE", "DETECTION_METHOD", "FEATURE_DECISION_REASON",
        "MULTI_FEATURE_CONTRIBUTOR_FLAG", "MULTI_FEATURE_CHANGES_FLAG",
        "MULTI_FEATURE_SCORE",
    ]
    for column in decision_columns:
        if column in result.columns:
            result[f"MODE1_{column}"] = result[column]
    result["MODE1_ECGI_DETECTED_FLAG"] = (
        result.groupby("ECGI")["REPORTABLE_EVIDENCE_ROW_FLAG"]
        .transform("max").fillna(0).astype(int)
    )
    return result


def _ecgi_if_admission(data: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    """Admit an ECGI when at least one genuinely scored feature crosses the Mode-2 tail."""
    header("PHASE 5A - MODE-2 ISOLATION-FOREST ADMISSION")
    result = data.copy()
    if result.empty:
        return result
    threshold = float(args.if_dominant_percentile)
    result["IF_DOMINANT_FEATURE_DETECTED_FLAG"] = (
        result["POPULATION"].eq("CANDIDATE")
        & col_num(result, "MODEL_AVAILABLE_FLAG", 0).eq(1)
        & col_num(result, "SOURCE_CHANGE_EVENTS", 0).gt(0)
        & col_num(result, "MODEL_NORMAL_PERCENTILE", 0).ge(threshold)
    ).astype(int)
    result["IF_DOMINANT_ECGI_DETECTED_FLAG"] = (
        result.groupby("ECGI")["IF_DOMINANT_FEATURE_DETECTED_FLAG"]
        .transform("max").fillna(0).astype(int)
    )
    detected_percentile = col_num(
        result, "MODEL_NORMAL_PERCENTILE", 0
    ).where(result["IF_DOMINANT_FEATURE_DETECTED_FLAG"].eq(1))
    result["IF_DOMINANT_MAX_PERCENTILE"] = (
        detected_percentile.groupby(result["ECGI"]).transform("max").fillna(0.0)
    )
    result["IF_DOMINANT_SELECTED_FEATURE_COUNT"] = (
        result["IF_DOMINANT_FEATURE_DETECTED_FLAG"]
        .groupby(result["ECGI"]).transform("sum").fillna(0).astype(int)
    )
    scored = (
        result["POPULATION"].eq("CANDIDATE")
        & col_num(result, "MODEL_AVAILABLE_FLAG", 0).eq(1)
        & col_num(result, "SOURCE_CHANGE_EVENTS", 0).gt(0)
    )
    passed = result["IF_DOMINANT_FEATURE_DETECTED_FLAG"].eq(1)
    admitted = result["IF_DOMINANT_ECGI_DETECTED_FLAG"].eq(1)
    scored_ecgi_max = (
        col_num(result, "MODEL_NORMAL_PERCENTILE", 0)
        .where(scored)
        .groupby(result["ECGI"]).max().dropna()
    )
    exact_100_ecgis = int(scored_ecgi_max.ge(100.0 - 1e-12).sum())
    log(
        f"IF threshold={threshold:.4f}; scored changed feature rows="
        f"{int(scored.sum()):,}; feature rows passing threshold="
        f"{int(passed.sum()):,}; ECGIs admitted={int(result.loc[admitted, 'ECGI'].nunique()):,}; "
        f"scored ECGIs with max percentile=100: {exact_100_ecgis:,}.",
        1,
    )
    return result


def _inspect_mode2_signatures(
    data: pd.DataFrame,
    events: Optional[pd.DataFrame],
    args: argparse.Namespace,
) -> pd.DataFrame:
    """Inspect source history only after IF admission and map business signatures.

    This is intentionally independent of ``assign_feature_clusters`` and every
    Mode-1/RAW decision flag.  Mode 2 starts from model scores, clusters the
    admitted anomalies, and only then explains those anomalies from their
    already-engineered temporal metrics and admitted change events.
    """
    result = data.copy()
    admitted = col_num(result, "IF_DOMINANT_ECGI_DETECTED_FLAG", 0).eq(1)
    recurrent = (
        col_num(result, "SOURCE_CHANGE_DAYS", 0).ge(2)
        & col_num(result, "SOURCE_CHANGE_EVENTS", 0).ge(2)
    )
    reversals = (
        col_num(result, "A_B_A_REVERT_COUNT", 0)
        + col_num(result, "REVERSE_TRANSITION_COUNT", 0)
    )
    result["MODE2_SIGNATURE_BACK_FORTH_FLAG"] = (
        admitted & recurrent & reversals.gt(0)
    ).astype(int)
    result["MODE2_SIGNATURE_HIGH_FREQUENCY_FLAG"] = (
        admitted
        & col_num(result, "SOURCE_CHANGE_DAYS", 0).ge(
            int(args.high_frequency_min_change_days)
        )
    ).astype(int)

    admitted_ecgis = set(
        result.loc[admitted, "ECGI"].dropna().astype(str)
    )
    if events is None or events.empty or not admitted_ecgis:
        admitted_events = pd.DataFrame()
    else:
        admitted_events = events.loc[
            events["POPULATION"].astype(str).eq("CANDIDATE")
            & events["ECGI"].astype(str).isin(admitted_ecgis)
        ].copy()
    # Rebuild topology from admitted histories only. This function consumes
    # source change events, not any Mode-1 decision.
    result = build_multi_feature_topology(result, admitted_events, args)
    # The topology merge may replace a non-consecutive source index with a new
    # RangeIndex. Rebuild the mask from the returned frame so boolean alignment
    # is correct for production data as well as compact unit fixtures.
    admitted = col_num(
        result, "IF_DOMINANT_ECGI_DETECTED_FLAG", 0
    ).eq(1)
    result["MODE2_SIGNATURE_MULTI_FEATURE_FLAG"] = (
        admitted
        & col_num(result, "MULTI_FEATURE_CHANGES_FLAG", 0).eq(1)
    ).astype(int)

    flags = result.loc[admitted].groupby("ECGI", as_index=True).agg(
        HAS_BACK_FORTH=("MODE2_SIGNATURE_BACK_FORTH_FLAG", "max"),
        HAS_HIGH_FREQUENCY=("MODE2_SIGNATURE_HIGH_FREQUENCY_FLAG", "max"),
        HAS_MULTI_FEATURE=("MODE2_SIGNATURE_MULTI_FEATURE_FLAG", "max"),
    )
    mapped = np.select(
        [
            col_num(flags, "HAS_BACK_FORTH", 0).eq(1),
            col_num(flags, "HAS_HIGH_FREQUENCY", 0).eq(1),
            col_num(flags, "HAS_MULTI_FEATURE", 0).eq(1),
        ],
        [
            "REAL_BACK_AND_FORTH",
            "HIGH_FREQUENCY_CHANGES",
            "MULTI_FEATURE_CHANGES",
        ],
        default="RARE_CHANGES",
    )
    mapping = pd.Series(mapped, index=flags.index)
    result["MODE2_MAPPED_CLUSTER"] = result["ECGI"].map(mapping).fillna("")

    # Compatibility aliases are regenerated from Mode-2 inspection so existing
    # audits remain readable; they are not inputs to this path.
    result["RAW_REAL_BACK_AND_FORTH_FLAG"] = result[
        "MODE2_SIGNATURE_BACK_FORTH_FLAG"
    ]
    result["RAW_HIGH_FREQUENCY_CHANGES_FLAG"] = result[
        "MODE2_SIGNATURE_HIGH_FREQUENCY_FLAG"
    ]
    result["RAW_BACKFORTH_AND_HIGH_FREQUENCY_OVERLAP_FLAG"] = (
        result["MODE2_SIGNATURE_BACK_FORTH_FLAG"].eq(1)
        & result["MODE2_SIGNATURE_HIGH_FREQUENCY_FLAG"].eq(1)
    ).astype(int)
    rare_mapped = (
        admitted
        & result["MODE2_MAPPED_CLUSTER"].eq("RARE_CHANGES")
        & col_num(result, "IF_DOMINANT_FEATURE_DETECTED_FLAG", 0).eq(1)
    )
    result["RAW_RARE_RECURRENT_MODEL_FLAG"] = (
        rare_mapped
        & col_num(result, "SOURCE_CHANGE_DAYS", 0).ge(2)
        & col_num(result, "SOURCE_CHANGE_EVENTS", 0).ge(2)
    ).astype(int)
    # Mode 2 intentionally uses its configured general IF percentile rather
    # than Mode 1's one-change conditional-conformal decision branch.
    result["RAW_RARE_ONE_CHANGE_CONDITIONAL_MODEL_FLAG"] = 0
    result["RAW_RARE_CONFLICT_FLAG"] = (
        admitted
        & (
            col_num(result, "CURRENT_DAY_SAME_DAY_CONFLICT_FLAG", 0).eq(1)
            | col_num(result, "RECURRENT_SAME_DAY_CONFLICT_FLAG", 0).eq(1)
        )
    ).astype(int)
    result["RAW_MULTI_FEATURE_TOPOLOGY_FLAG"] = col_num(
        result, "RAW_MULTI_FEATURE_TOPOLOGY_FLAG", 0
    ).astype(int)
    result["RAW_QUALIFIED_FEATURE_FLAG"] = (
        result["MODE2_SIGNATURE_BACK_FORTH_FLAG"].eq(1)
        | result["MODE2_SIGNATURE_HIGH_FREQUENCY_FLAG"].eq(1)
        | result["MODE2_SIGNATURE_MULTI_FEATURE_FLAG"].eq(1)
        | col_num(result, "IF_DOMINANT_FEATURE_DETECTED_FLAG", 0).eq(1)
    ).astype(int)
    return result


def _mode2_ecgi_vectors(data: pd.DataFrame) -> pd.DataFrame:
    """Return one deterministic temporal vector per IF-admitted ECGI."""
    detected = data.loc[
        col_num(data, "IF_DOMINANT_FEATURE_DETECTED_FLAG", 0).eq(1)
    ].copy()
    if detected.empty:
        return pd.DataFrame()
    matrix = prepare_model_matrix(detected)
    matrix.insert(0, "ECGI", detected["ECGI"].astype(str).to_numpy())
    vectors = matrix.groupby("ECGI", as_index=True).agg(
        {column: ["max", "mean"] for column in MODEL_INPUT_COLUMNS}
    )
    vectors.columns = [
        f"{column}_{stat}".upper() for column, stat in vectors.columns
    ]
    return vectors.sort_index()


def _choose_kmeans_k(
    scaled: np.ndarray,
    requested_k: int,
    automatic: bool,
    auto_max: int,
    random_state: int,
) -> Tuple[int, float, str]:
    """Clamp K safely and optionally select it by deterministic silhouette score."""
    row_count = int(len(scaled))
    unique_count = int(len(np.unique(np.asarray(scaled), axis=0))) if row_count else 0
    maximum = min(row_count, unique_count)
    if maximum <= 1:
        return 1, np.nan, "SINGLETON_OR_IDENTICAL_VECTORS"
    if not automatic:
        selected = max(1, min(int(requested_k), maximum))
        status = (
            "FIXED_REQUESTED"
            if selected == int(requested_k)
            else "FIXED_CLAMPED_TO_AVAILABLE_VECTORS"
        )
        return selected, np.nan, status
    try:
        from sklearn.cluster import KMeans
        from sklearn.metrics import silhouette_score
    except Exception as exc:
        raise RuntimeError(
            "Automatic KMeans selection requires scikit-learn."
        ) from exc
    upper = min(int(auto_max), maximum - 1)
    if upper < 2:
        return 1, np.nan, "AUTO_NO_VALID_SILHOUETTE_K"
    best_k, best_score = 2, -np.inf
    for k in range(2, upper + 1):
        labels = KMeans(
            n_clusters=k, n_init=10, random_state=int(random_state)
        ).fit_predict(scaled)
        if len(np.unique(labels)) < 2:
            continue
        score = float(silhouette_score(scaled, labels))
        if score > best_score + 1e-12:
            best_k, best_score = k, score
    return best_k, best_score, "AUTO_SILHOUETTE"


def _add_kmeans_diagnostics(
    data: pd.DataFrame,
    args: argparse.Namespace,
) -> pd.DataFrame:
    """Cluster IF-admitted ECGIs before rule explanation."""
    result = data.copy()
    result["KMEANS_CLUSTER"] = pd.Series(pd.NA, index=result.index, dtype="Int64")
    result["KMEANS_DISTANCE_TO_CENTROID"] = np.nan
    result["KMEANS_SELECTED_K"] = 0
    result["KMEANS_SELECTION_STATUS"] = "NO_IF_DOMINANT_ECGIS"
    result["KMEANS_SILHOUETTE"] = np.nan
    vectors = _mode2_ecgi_vectors(result)
    if vectors.empty:
        return result
    try:
        from sklearn.cluster import KMeans
        from sklearn.preprocessing import StandardScaler
    except Exception as exc:
        raise RuntimeError(
            "Mode 2/3 KMeans diagnostics require scikit-learn."
        ) from exc
    scaled = StandardScaler().fit_transform(vectors.to_numpy(dtype=float))
    k, silhouette, status = _choose_kmeans_k(
        scaled,
        int(args.kmeans_clusters),
        bool(args.kmeans_auto),
        int(args.kmeans_auto_max),
        int(args.random_state),
    )
    if k == 1:
        labels = np.zeros(len(vectors), dtype=int)
        distances = np.linalg.norm(scaled - scaled.mean(axis=0), axis=1)
    else:
        model = KMeans(
            n_clusters=int(k), n_init=10, random_state=int(args.random_state)
        )
        labels = model.fit_predict(scaled)
        distances = np.linalg.norm(scaled - model.cluster_centers_[labels], axis=1)
    cluster_map = dict(zip(vectors.index.astype(str), labels.astype(int)))
    distance_map = dict(zip(vectors.index.astype(str), distances.astype(float)))
    ecgi = result["ECGI"].astype(str)
    result["KMEANS_CLUSTER"] = ecgi.map(cluster_map).astype("Int64")
    result["KMEANS_DISTANCE_TO_CENTROID"] = ecgi.map(distance_map)
    result.loc[ecgi.isin(cluster_map), "KMEANS_SELECTED_K"] = int(k)
    result.loc[ecgi.isin(cluster_map), "KMEANS_SELECTION_STATUS"] = status
    result.loc[ecgi.isin(cluster_map), "KMEANS_SILHOUETTE"] = silhouette
    return result


def _add_kmeans_rule_compatibility(
    data: pd.DataFrame,
    args: argparse.Namespace,
) -> pd.DataFrame:
    """Compare unsupervised groups with post-model signature labels."""
    result = data.copy()
    if bool(args.kmeans_rule_compatibility):
        admitted = result.loc[
            col_num(result, "IF_DOMINANT_ECGI_DETECTED_FLAG", 0).eq(1),
            ["ECGI", "KMEANS_CLUSTER", "MODE2_MAPPED_CLUSTER"],
        ].drop_duplicates("ECGI")
        if admitted.empty:
            return result
        priority = {
            "REAL_BACK_AND_FORTH": 4,
            "HIGH_FREQUENCY_CHANGES": 3,
            "MULTI_FEATURE_CHANGES": 2,
            "RARE_CHANGES": 1,
        }
        counts = (
            admitted.groupby(["KMEANS_CLUSTER", "MODE2_MAPPED_CLUSTER"], dropna=False)
            .size().rename("N").reset_index()
        )
        counts["_PRIORITY"] = counts["MODE2_MAPPED_CLUSTER"].map(priority).fillna(0)
        majority = (
            counts.sort_values(
                ["KMEANS_CLUSTER", "N", "_PRIORITY", "MODE2_MAPPED_CLUSTER"],
                ascending=[True, False, False, True],
            )
            .groupby("KMEANS_CLUSTER", as_index=False).head(1)
            .set_index("KMEANS_CLUSTER")["MODE2_MAPPED_CLUSTER"]
        )
        cluster_size = admitted.groupby("KMEANS_CLUSTER").size()
        majority_count = (
            counts.sort_values(
                ["KMEANS_CLUSTER", "N", "_PRIORITY", "MODE2_MAPPED_CLUSTER"],
                ascending=[True, False, False, True],
            )
            .groupby("KMEANS_CLUSTER", as_index=False).head(1)
            .set_index("KMEANS_CLUSTER")["N"]
        )
        result["KMEANS_MAJORITY_RULE_CLUSTER"] = result["KMEANS_CLUSTER"].map(majority)
        result["KMEANS_CLUSTER_SIZE"] = result["KMEANS_CLUSTER"].map(cluster_size)
        result["KMEANS_RULE_PURITY"] = (
            result["KMEANS_CLUSTER"].map(majority_count)
            / result["KMEANS_CLUSTER"].map(cluster_size).replace(0, np.nan)
        )
        result["KMEANS_RULE_COMPATIBILITY"] = np.where(
            col_num(result, "IF_DOMINANT_ECGI_DETECTED_FLAG", 0).eq(0),
            "",
            np.where(
                result["MODE2_MAPPED_CLUSTER"].eq(
                    result["KMEANS_MAJORITY_RULE_CLUSTER"]
                ),
                "COMPATIBLE",
                "DIFFERENT_FROM_KMEANS_MAJORITY",
            ),
        )
    return result


def _build_mode2_view(
    data: pd.DataFrame,
    args: argparse.Namespace,
    events: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """Run IF-first Mode 2 and project it into the V9.8.4 summary contract."""
    if data is None or data.empty:
        return pd.DataFrame() if data is None else data.copy()
    reusable = all(
        column in data.columns
        for column in [
            "IF_DOMINANT_FEATURE_DETECTED_FLAG",
            "IF_DOMINANT_ECGI_DETECTED_FLAG",
            "IF_DOMINANT_MAX_PERCENTILE",
            "IF_DOMINANT_SELECTED_FEATURE_COUNT",
            "MODE2_MAPPED_CLUSTER",
            "KMEANS_CLUSTER",
            "MODE2_SIGNATURE_BACK_FORTH_FLAG",
            "MODE2_SIGNATURE_HIGH_FREQUENCY_FLAG",
            "MODE2_SIGNATURE_MULTI_FEATURE_FLAG",
            "MODE2_ECGI_RETAINED_FLAG",
            "MODE2_INDEPENDENT_ONE_CHANGE_EXTREME_FLAG",
        ]
    )
    if reusable:
        result = data.copy()
    else:
        result = _ecgi_if_admission(data, args)
        # ML-first order: admission -> unsupervised grouping -> explanation.
        result = _add_kmeans_diagnostics(result, args)
        result = _inspect_mode2_signatures(result, events, args)
        result = _add_kmeans_rule_compatibility(result, args)
    admitted = col_num(result, "IF_DOMINANT_ECGI_DETECTED_FLAG", 0).eq(1)
    recurrent_feature = (
        col_num(result, "SOURCE_CHANGE_DAYS", 0).ge(2)
        & col_num(result, "SOURCE_CHANGE_EVENTS", 0).ge(2)
    )
    objective_conflict = (
        col_num(result, "CURRENT_DAY_SAME_DAY_CONFLICT_FLAG", 0).eq(1)
        | col_num(result, "RECURRENT_SAME_DAY_CONFLICT_FLAG", 0).eq(1)
    )
    independent_one_change_feature = (
        admitted
        & col_num(result, "IF_DOMINANT_FEATURE_DETECTED_FLAG", 0).eq(1)
        & col_num(result, "SOURCE_CHANGE_DAYS", 0).eq(1)
        & col_num(result, "SOURCE_CHANGE_EVENTS", 0).eq(1)
        & col_num(
            result, "MODEL_ONE_CHANGE_CONDITIONAL_AVAILABLE_FLAG", 0
        ).eq(1)
        & col_num(result, "MODEL_ONE_CHANGE_CONFORMAL_P_MAX", 1.0).le(
            float(args.one_change_model_alpha)
        )
        & col_num(result, "OBSERVED_COVERAGE_RATE", 0).ge(
            float(args.one_change_min_coverage)
        )
        & col_num(
            result, "LATEST_OBSERVED_ON_END_DATE_FLAG", 0
        ).eq(1)
    )
    result["MODE2_INDEPENDENT_ONE_CHANGE_EXTREME_FEATURE_FLAG"] = (
        independent_one_change_feature.astype(int)
    )
    independent_one_change = (
        independent_one_change_feature.groupby(result["ECGI"]).transform("max")
        & admitted
    )
    result["MODE2_INDEPENDENT_ONE_CHANGE_EXTREME_FLAG"] = independent_one_change.astype(int)
    structural_or_recurrent = (
        col_num(result, "MODE2_SIGNATURE_BACK_FORTH_FLAG", 0).eq(1)
        | col_num(result, "MODE2_SIGNATURE_HIGH_FREQUENCY_FLAG", 0).eq(1)
        | col_num(result, "MODE2_SIGNATURE_MULTI_FEATURE_FLAG", 0).eq(1)
        | recurrent_feature
        | objective_conflict
        | independent_one_change
    )
    retained_by_ecgi = (
        structural_or_recurrent.groupby(result["ECGI"]).transform("max")
        & admitted
    )
    result["MODE2_ECGI_RETAINED_FLAG"] = retained_by_ecgi.astype(int)
    raw_admitted_ecgis = int(result.loc[admitted, "ECGI"].nunique())
    retained_ecgis = int(result.loc[retained_by_ecgi, "ECGI"].nunique())
    independent_ecgis = int(
        result.loc[independent_one_change, "ECGI"].nunique()
    )
    log(
        f"Mode-2 refinement retained {retained_ecgis:,}/{raw_admitted_ecgis:,} "
        f"IF-admitted ECGIs; independent extreme one-change ECGIs={independent_ecgis:,}; "
        f"ordinary one-time RARE removed={max(0, raw_admitted_ecgis-retained_ecgis):,}.",
        1,
    )
    admitted = retained_by_ecgi
    mapped_cluster = result["MODE2_MAPPED_CLUSTER"]
    evidence = (
        admitted
        & (
            (
                mapped_cluster.eq("REAL_BACK_AND_FORTH")
                & col_num(result, "MODE2_SIGNATURE_BACK_FORTH_FLAG", 0).eq(1)
            )
            | (
                mapped_cluster.eq("HIGH_FREQUENCY_CHANGES")
                & col_num(result, "MODE2_SIGNATURE_HIGH_FREQUENCY_FLAG", 0).eq(1)
            )
            | (
                mapped_cluster.eq("MULTI_FEATURE_CHANGES")
                & col_num(result, "MULTI_FEATURE_CONTRIBUTOR_FLAG", 0).eq(1)
            )
            | (
                mapped_cluster.eq("RARE_CHANGES")
                & col_num(result, "IF_DOMINANT_FEATURE_DETECTED_FLAG", 0).eq(1)
            )
        )
    )
    # A retained ECGI always keeps at least one auditable evidence row.
    has_evidence = evidence.groupby(result["ECGI"]).transform("max")
    evidence = evidence | (
        admitted & ~has_evidence
        & col_num(result, "IF_DOMINANT_FEATURE_DETECTED_FLAG", 0).eq(1)
    )
    result["FEATURE_REPORTABLE_FLAG"] = evidence.astype(int)
    result["REPORTABLE_EVIDENCE_ROW_FLAG"] = evidence.astype(int)
    result["FEATURE_FINAL_CLUSTER"] = np.where(evidence, mapped_cluster, "")
    result["FEATURE_CLUSTER_SCORE"] = np.where(
        evidence,
        col_num(result, "IF_DOMINANT_MAX_PERCENTILE", 0),
        0.0,
    )
    result["REAL_BACK_AND_FORTH_FLAG"] = (
        evidence & mapped_cluster.eq("REAL_BACK_AND_FORTH")
    ).astype(int)
    result["HIGH_FREQUENCY_CHANGES_FLAG"] = (
        evidence & mapped_cluster.eq("HIGH_FREQUENCY_CHANGES")
    ).astype(int)
    result["RARE_CHANGES_FLAG"] = (
        evidence & mapped_cluster.eq("RARE_CHANGES")
    ).astype(int)
    result["ONE_DAY_RARE_GUARD_FLAG"] = (
        col_num(result, "IF_DOMINANT_ECGI_DETECTED_FLAG", 0).eq(1)
        & col_num(result, "MODE2_ECGI_RETAINED_FLAG", 0).eq(0)
        & col_num(result, "SOURCE_CHANGE_DAYS", 0).le(1)
    ).astype(int)
    result["MULTI_FEATURE_CHANGES_FLAG"] = (
        admitted & mapped_cluster.eq("MULTI_FEATURE_CHANGES")
    ).astype(int)
    result["MULTI_FEATURE_CONTRIBUTOR_FLAG"] = (
        result["MULTI_FEATURE_CHANGES_FLAG"].eq(1)
        & col_num(result, "MULTI_FEATURE_CONTRIBUTOR_FLAG", 0).eq(1)
    ).astype(int)
    result["MULTI_FEATURE_SCORE"] = np.where(
        result["MULTI_FEATURE_CHANGES_FLAG"].eq(1),
        col_num(result, "IF_DOMINANT_MAX_PERCENTILE", 0),
        0.0,
    )
    result["EVIDENCE_ROLE"] = np.where(
        evidence,
        np.where(
            mapped_cluster.eq("MULTI_FEATURE_CHANGES"),
            "MULTI_FEATURE_TOPOLOGY_CONTRIBUTOR",
            "IF_DOMINANT_SIGNATURE_MAPPED_FEATURE",
        ),
        "NONE",
    )
    result["DETECTION_METHOD"] = np.where(
        evidence,
        "IF_PERCENTILE_ADMISSION_THEN_SIGNATURE_MAPPING",
        "NONE",
    )
    result["FEATURE_DECISION_REASON"] = np.where(
        evidence,
        (
            "ECGI admitted by Isolation Forest percentile >= "
            f"{float(args.if_dominant_percentile):.4f}; full source history "
            "mapped it to the structural signature cluster"
        ),
        np.where(
            col_num(result, "IF_DOMINANT_ECGI_DETECTED_FLAG", 0).eq(1)
            & col_num(result, "MODE2_ECGI_RETAINED_FLAG", 0).eq(0),
            "IF-admitted ordinary one-time change did not pass the independent "
            "same-age one-change conformal test",
            "Not admitted by the Mode-2 Isolation Forest percentile gate",
        ),
    )
    result["DETECTION_MODE_TAG"] = np.where(admitted, MODE2_LABEL, "")
    quiet_column = (
        "QUIET_LAST_N_DAYS_FLAG"
        if "QUIET_LAST_N_DAYS_FLAG" in result.columns
        else "STABLE_LAST_N_DAYS_FLAG"
    )
    result["WOULD_OLD_STABILITY_OVERRIDE_DROP_FLAG"] = (
        admitted
        & col_num(result, quiet_column, 0).eq(1)
        & result["REPORTABLE_EVIDENCE_ROW_FLAG"].eq(1)
    ).astype(int)
    result["DROPPED_BY_RECENCY_FLAG"] = 0
    return result


def finalize_detection(
    data: pd.DataFrame,
    events: pd.DataFrame,
    args: argparse.Namespace,
) -> pd.DataFrame:
    """Run the requested detector; Mode 1 and Mode 2 are independent branches."""
    if args.detection_mode == "if_dominant":
        return _build_mode2_view(data, args, events)

    if args.detection_mode == "signature_dominant":
        mode1 = _finalize_signature_dominant(data, events, args)
        mode1["DETECTION_MODE_TAG"] = MODE1_LABEL
        mode1["MODE1_ECGI_DETECTED_FLAG"] = (
            mode1.groupby("ECGI")["REPORTABLE_EVIDENCE_ROW_FLAG"]
            .transform("max").fillna(0).astype(int)
        ) if not mode1.empty else pd.Series(dtype=int)
        return mode1
    # Compare mode evaluates both branches independently from the same scored
    # raw dataset, then carries Mode-1 fields beside the Mode-2 projection.
    # A stable row id prevents either branch's topology merge from silently
    # aligning decisions by a newly-created pandas index.
    row_id = "_MODE_COMPARISON_ROW_ID"
    compare_input = data.copy()
    compare_input[row_id] = np.arange(len(compare_input), dtype=np.int64)
    mode1 = _finalize_signature_dominant(compare_input, events, args)
    preserved = _copy_mode1_decisions(mode1)
    mode2 = _build_mode2_view(compare_input, args, events)
    mode1_columns = [
        column for column in preserved.columns if column.startswith("MODE1_")
    ]
    mode2 = mode2.merge(
        preserved[[row_id, *mode1_columns]],
        on=row_id,
        how="left",
        validate="one_to_one",
        sort=False,
    )
    # Compare mode retains the Mode-1 reporting projection while carrying every
    # Mode-2 field. The comparison summary performs the one-row ECGI union.
    for column in [
        "FEATURE_REPORTABLE_FLAG", "FEATURE_FINAL_CLUSTER",
        "FEATURE_CLUSTER_SCORE", "REPORTABLE_EVIDENCE_ROW_FLAG",
        "EVIDENCE_ROLE", "DETECTION_METHOD", "FEATURE_DECISION_REASON",
        "MULTI_FEATURE_CONTRIBUTOR_FLAG", "MULTI_FEATURE_CHANGES_FLAG",
        "MULTI_FEATURE_SCORE",
    ]:
        mode1_column = f"MODE1_{column}"
        if mode1_column in mode2.columns:
            mode2[column] = mode2[mode1_column]
    mode2["DETECTION_MODE_TAG"] = np.select(
        [
            mode2["MODE1_ECGI_DETECTED_FLAG"].eq(1)
            & mode2["IF_DOMINANT_ECGI_DETECTED_FLAG"].eq(1),
            mode2["MODE1_ECGI_DETECTED_FLAG"].eq(1),
            mode2["IF_DOMINANT_ECGI_DETECTED_FLAG"].eq(1),
        ],
        [f"{MODE1_LABEL}; {MODE2_LABEL}", MODE1_LABEL, MODE2_LABEL],
        default="",
    )
    mode2.drop(columns=[row_id], inplace=True)
    return mode2


def _build_signature_dominant_summary(data: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    header("PHASE 6 - ONE FINAL FOUR-CLUSTER DECISION PER ECGI")
    if data is None or data.empty:
        return pd.DataFrame()
    data = data.copy()
    if "REPORTABLE_EVIDENCE_ROW_FLAG" not in data.columns:
        data["REPORTABLE_EVIDENCE_ROW_FLAG"] = (
            col_num(data, "FEATURE_REPORTABLE_FLAG", 0).eq(1)
            | col_num(data, "MULTI_FEATURE_CONTRIBUTOR_FLAG", 0).eq(1)
        ).astype(int)
    if "EVIDENCE_ROLE" not in data.columns:
        data["EVIDENCE_ROLE"] = np.select(
            [
                col_num(data, "FEATURE_REPORTABLE_FLAG", 0).eq(1)
                & col_num(data, "MULTI_FEATURE_CONTRIBUTOR_FLAG", 0).eq(1),
                col_num(data, "FEATURE_REPORTABLE_FLAG", 0).eq(1),
                col_num(data, "MULTI_FEATURE_CONTRIBUTOR_FLAG", 0).eq(1),
            ],
            ["BOTH", "INDIVIDUAL_FEATURE_PATTERN", "MULTI_FEATURE_TOPOLOGY_CONTRIBUTOR"],
            default="NONE",
        )
    feature_candidates = data[
        data["POPULATION"].eq("CANDIDATE")
        & col_num(data, "FEATURE_REPORTABLE_FLAG", 0).eq(1)
        & data["FEATURE_FINAL_CLUSTER"].isin(FINAL_CLUSTERS)
    ].copy()
    feature_scores = (
        feature_candidates.groupby(["ECGI", "FEATURE_FINAL_CLUSTER"], as_index=False)
        .agg(CLUSTER_SCORE=("FEATURE_CLUSTER_SCORE", "max"))
        .rename(columns={"FEATURE_FINAL_CLUSTER": "CLUSTER_CANDIDATE"})
    ) if not feature_candidates.empty else pd.DataFrame(
        columns=["ECGI", "CLUSTER_CANDIDATE", "CLUSTER_SCORE"]
    )
    multi_rows = data[
        data["POPULATION"].eq("CANDIDATE")
        & col_num(data, "MULTI_FEATURE_CHANGES_FLAG", 0).eq(1)
    ][["ECGI", "MULTI_FEATURE_SCORE"]].drop_duplicates("ECGI")
    if not multi_rows.empty:
        multi_scores = multi_rows.rename(columns={"MULTI_FEATURE_SCORE": "CLUSTER_SCORE"})
        multi_scores["CLUSTER_CANDIDATE"] = "MULTI_FEATURE_CHANGES"
        multi_scores = multi_scores[["ECGI", "CLUSTER_CANDIDATE", "CLUSTER_SCORE"]]
    else:
        multi_scores = pd.DataFrame(columns=["ECGI", "CLUSTER_CANDIDATE", "CLUSTER_SCORE"])
    cluster_scores = pd.concat([feature_scores, multi_scores], ignore_index=True)
    if cluster_scores.empty:
        log("No ECGI qualified for the four final clusters.", 1)
        return pd.DataFrame()
    cluster_scores["TIE_PRIORITY"] = cluster_scores["CLUSTER_CANDIDATE"].map(
        CLUSTER_TIE_PRIORITY
    ).fillna(0)
    selected = (
        cluster_scores.sort_values(
            ["ECGI", "CLUSTER_SCORE", "TIE_PRIORITY"],
            ascending=[True, False, False],
        )
        .groupby("ECGI", as_index=False).head(1)
        .rename(columns={
            "CLUSTER_CANDIDATE": "PRIMARY_ANOMALY_CLUSTER",
            "CLUSTER_SCORE": "ECGI_ANOMALY_RISK_SCORE",
        })
    )
    all_matches = (
        cluster_scores.sort_values(["ECGI", "TIE_PRIORITY"], ascending=[True, False])
        .groupby("ECGI")["CLUSTER_CANDIDATE"]
        .agg(lambda values: ";".join(dict.fromkeys(map(str, values))))
        .rename("ALL_MATCHED_CLUSTERS").reset_index()
    )
    match_counts = cluster_scores.groupby("ECGI")["CLUSTER_CANDIDATE"].nunique().rename(
        "MATCHED_CLUSTER_COUNT"
    ).reset_index()
    selected = selected.merge(all_matches, on="ECGI", how="left").merge(
        match_counts, on="ECGI", how="left"
    )
    selected_cluster = selected.set_index("ECGI")["PRIMARY_ANOMALY_CLUSTER"].to_dict()
    evidence = data[
        data["POPULATION"].eq("CANDIDATE")
        & col_num(data, "REPORTABLE_EVIDENCE_ROW_FLAG", 0).eq(1)
        & data["ECGI"].isin(selected["ECGI"])
    ].copy()
    evidence["SELECTED_CLUSTER"] = evidence["ECGI"].map(selected_cluster)
    evidence["PRIMARY_MATCH_FLAG"] = np.where(
        evidence["SELECTED_CLUSTER"].eq("MULTI_FEATURE_CHANGES"),
        col_num(evidence, "MULTI_FEATURE_CONTRIBUTOR_FLAG", 0).eq(1),
        evidence["FEATURE_FINAL_CLUSTER"].eq(evidence["SELECTED_CLUSTER"]),
    ).astype(int)
    evidence = evidence.sort_values(
        ["ECGI", "PRIMARY_MATCH_FLAG", "SOURCE_CHANGE_DAYS", "FEATURE_CLUSTER_SCORE",
         "MODEL_NORMAL_PERCENTILE", "FEATURE"],
        ascending=[True, False, False, False, False, True],
    )
    primary = evidence.groupby("ECGI", as_index=False).head(1).copy()
    primary_keep = primary[[
        "ECGI", "FEATURE", "FEATURE_FINAL_CLUSTER", "EVIDENCE_ROLE",
        "FEATURE_CLUSTER_SCORE", "SOURCE_CHANGE_DAYS", "SOURCE_CHANGE_EVENTS",
        "RECENT_CHANGE_DAYS", "A_B_A_REVERT_COUNT", "REVERSE_TRANSITION_COUNT",
        "MODEL_NORMAL_PERCENTILE", "DETECTION_METHOD", "LATEST_SOURCE_VALUE",
        "PREVIOUS_DIFFERENT_SOURCE_VALUE", "FIRST_CHANGE_DATE", "LAST_CHANGE_DATE",
        "FEATURE_DECISION_REASON",
    ]].rename(columns={
        "FEATURE": "PRIMARY_EVIDENCE_FEATURE",
        "FEATURE_FINAL_CLUSTER": "PRIMARY_EVIDENCE_FEATURE_CLUSTER",
        "EVIDENCE_ROLE": "PRIMARY_EVIDENCE_ROLE",
        "FEATURE_CLUSTER_SCORE": "PRIMARY_FEATURE_SCORE",
        "SOURCE_CHANGE_DAYS": "PRIMARY_FEATURE_CHANGE_DAYS",
        "SOURCE_CHANGE_EVENTS": "PRIMARY_FEATURE_CHANGE_EVENTS",
        "RECENT_CHANGE_DAYS": "PRIMARY_FEATURE_RECENT_CHANGE_DAYS",
        "A_B_A_REVERT_COUNT": "PRIMARY_FEATURE_ABA_REVERT_COUNT",
        "REVERSE_TRANSITION_COUNT": "PRIMARY_FEATURE_REVERSE_TRANSITION_COUNT",
        "MODEL_NORMAL_PERCENTILE": "PRIMARY_FEATURE_MODEL_NORMAL_PERCENTILE",
        "DETECTION_METHOD": "PRIMARY_DETECTION_METHOD",
        "LATEST_SOURCE_VALUE": "PRIMARY_LATEST_SOURCE_VALUE",
        "PREVIOUS_DIFFERENT_SOURCE_VALUE": "PRIMARY_PREVIOUS_DIFFERENT_SOURCE_VALUE",
        "FIRST_CHANGE_DATE": "PRIMARY_FIRST_CHANGE_DATE",
        "LAST_CHANGE_DATE": "PRIMARY_LAST_CHANGE_DATE",
        "FEATURE_DECISION_REASON": "PRIMARY_FEATURE_DECISION_REASON",
    })
    candidate_rows = data[data["POPULATION"].eq("CANDIDATE")].copy()
    if "QUIET_LAST_N_DAYS_FLAG" not in candidate_rows.columns:
        candidate_rows["QUIET_LAST_N_DAYS_FLAG"] = col_num(
            candidate_rows, "STABLE_LAST_N_DAYS_FLAG", 0
        )
    if "WOULD_OLD_STABILITY_OVERRIDE_DROP_FLAG" not in candidate_rows.columns:
        candidate_rows["WOULD_OLD_STABILITY_OVERRIDE_DROP_FLAG"] = 0
    for column, default in {
        "REGION": "", "MARKETCLUSTER": "", "MARKET": "", "SUBMARKET": "",
        "SERVER_ID": "", "USID": "", "SITE_TYPE": "UNKNOWN",
        "SITE_TYPE_MATCH_STATUS": "UNMATCHED", "IS_COW_SITE": 0,
    }.items():
        if column not in candidate_rows.columns:
            candidate_rows[column] = default
    meta = candidate_rows.groupby("ECGI", as_index=False).agg(
        REGION=("REGION", "last"),
        MARKETCLUSTER=("MARKETCLUSTER", "last"),
        MARKET=("MARKET", "last"),
        SUBMARKET=("SUBMARKET", "last"),
        SERVER_ID=("SERVER_ID", "last"),
        USID=("USID", "last"),
        SITE_TYPE=("SITE_TYPE", "last"),
        SITE_TYPE_MATCH_STATUS=("SITE_TYPE_MATCH_STATUS", "last"),
        IS_COW_SITE=("IS_COW_SITE", "max"),
        QUIET_LAST_N_DAYS_FLAG=("QUIET_LAST_N_DAYS_FLAG", "max"),
        STABLE_LAST_N_DAYS_FLAG=("STABLE_LAST_N_DAYS_FLAG", "max"),
        WOULD_OLD_STABILITY_OVERRIDE_DROP_FLAG=(
            "WOULD_OLD_STABILITY_OVERRIDE_DROP_FLAG", "max"
        ),
        STABILITY_STATUS=("STABILITY_STATUS", "last"),
        ECGI_RECENT_OBSERVED_DAYS=("ECGI_RECENT_OBSERVED_DAYS", "max"),
        ECGI_RECENT_CHANGE_DAYS=("ECGI_RECENT_CHANGE_DAYS", "max"),
        ECGI_DAYS_SINCE_LAST_ANY_CHANGE=("ECGI_DAYS_SINCE_LAST_ANY_CHANGE", "min"),
        MULTI_CHANGED_FEATURE_COUNT=("MULTI_CHANGED_FEATURE_COUNT", "max"),
        MULTI_UNIQUE_CHANGE_DATES=("MULTI_UNIQUE_CHANGE_DATES", "max"),
        MULTI_FEATURE_DATE_EVENTS=("MULTI_FEATURE_DATE_EVENTS", "max"),
        MULTI_MAX_FEATURES_ON_ONE_DATE=("MULTI_MAX_FEATURES_ON_ONE_DATE", "max"),
        MULTI_MAX_SAME_DATE_FEATURE_SHARE=("MULTI_MAX_SAME_DATE_FEATURE_SHARE", "max"),
        MULTI_DISTINCT_FEATURE_DATE_PATTERNS=("MULTI_DISTINCT_FEATURE_DATE_PATTERNS", "max"),
        MULTI_FEATURE_CHANGES_FLAG=("MULTI_FEATURE_CHANGES_FLAG", "max"),
        MULTI_FEATURE_SCORE=("MULTI_FEATURE_SCORE", "max"),
    )
    report_counts = evidence.groupby("ECGI", as_index=False).agg(
        REPORTABLE_EVIDENCE_FEATURE_COUNT=("FEATURE", "nunique"),
        REPORTABLE_FEATURE_CHANGE_DAYS_SUM=("SOURCE_CHANGE_DAYS", "sum"),
        MAX_FEATURE_CHANGE_DAYS=("SOURCE_CHANGE_DAYS", "max"),
        MAX_MODEL_NORMAL_PERCENTILE=("MODEL_NORMAL_PERCENTILE", "max"),
    )
    summary = (
        selected[[
            "ECGI", "PRIMARY_ANOMALY_CLUSTER", "ECGI_ANOMALY_RISK_SCORE",
            "ALL_MATCHED_CLUSTERS", "MATCHED_CLUSTER_COUNT",
        ]]
        .merge(meta, on="ECGI", how="left")
        .merge(primary_keep, on="ECGI", how="left")
        .merge(report_counts, on="ECGI", how="left")
    )
    top_features = (
        evidence.sort_values(
            ["ECGI", "PRIMARY_MATCH_FLAG", "FEATURE_CLUSTER_SCORE", "SOURCE_CHANGE_DAYS"],
            ascending=[True, False, False, False],
        )
        .groupby("ECGI").head(int(args.summary_top_features))
        .groupby("ECGI")
        .apply(lambda frame: "; ".join(
            f"{row.FEATURE} (role={row.EVIDENCE_ROLE}, "
            f"cluster={row.FEATURE_FINAL_CLUSTER or 'MULTI_CONTRIBUTOR'}, "
            f"change_days={scalar_int(row.SOURCE_CHANGE_DAYS)}, "
            f"model_pctl={scalar_float(row.MODEL_NORMAL_PERCENTILE):.1f})"
            for row in frame.itertuples()
        ))
        .rename("TOP_EVIDENCE_FEATURES").reset_index()
    )
    summary = summary.merge(top_features, on="ECGI", how="left")
    summary["PRIMARY_ABNORMAL_FEATURE"] = summary["PRIMARY_EVIDENCE_FEATURE"].fillna("")
    summary["COW_REVIEW_NOTE"] = np.where(
        col_num(summary, "IS_COW_SITE", 0).eq(1),
        "COW SITE: frequent/back-and-forth movement may be operationally normal; SITE_TYPE was context-only and did not alter this result.",
        "",
    )
    summary["ECGI_LABEL"] = summary["ECGI"].map(lambda value: "ECGI_" + norm_key(value))
    summary["PLAIN_ENGLISH_SUMMARY"] = summary.apply(
        lambda row: (
            f"{row.get('ECGI_LABEL','')} matched {row.get('PRIMARY_ANOMALY_CLUSTER','')} "
            f"with score {scalar_float(row.get('ECGI_ANOMALY_RISK_SCORE',0)):.1f}. "
            f"The cell changed on {scalar_int(row.get('MULTI_UNIQUE_CHANGE_DATES',0))} distinct date(s) "
            f"across {scalar_int(row.get('MULTI_CHANGED_FEATURE_COUNT',0))} monitored feature(s). "
            f"SITE_TYPE={row.get('SITE_TYPE','UNKNOWN')} is review context only."
        ),
        axis=1,
    )
    summary = summary.sort_values(
        ["ECGI_ANOMALY_RISK_SCORE", "MATCHED_CLUSTER_COUNT", "MAX_FEATURE_CHANGE_DAYS", "ECGI"],
        ascending=[False, False, False, True],
    ).reset_index(drop=True)
    summary.insert(0, "INCIDENT_RANK", np.arange(1, len(summary) + 1))
    if summary["ECGI"].duplicated().any():
        raise AssertionError("Duplicate ECGI rows reached the final incident summary.")
    if not set(summary["PRIMARY_ANOMALY_CLUSTER"]).issubset(set(FINAL_CLUSTERS)):
        raise AssertionError("A final cluster outside the requested four was created.")
    log(
        f"Final ECGI incidents={len(summary):,}; clusters="
        + str(summary["PRIMARY_ANOMALY_CLUSTER"].value_counts().to_dict()),
        1,
    )
    return summary


def _restore_mode1_view(data: pd.DataFrame) -> pd.DataFrame:
    result = data.copy()
    for column in [
        "FEATURE_REPORTABLE_FLAG", "FEATURE_FINAL_CLUSTER",
        "FEATURE_CLUSTER_SCORE", "REPORTABLE_EVIDENCE_ROW_FLAG",
        "EVIDENCE_ROLE", "DETECTION_METHOD", "FEATURE_DECISION_REASON",
        "MULTI_FEATURE_CONTRIBUTOR_FLAG", "MULTI_FEATURE_CHANGES_FLAG",
        "MULTI_FEATURE_SCORE",
    ]:
        saved = f"MODE1_{column}"
        if saved in result.columns:
            result[column] = result[saved]
    return result


def _ecgi_kmeans_audit(data: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    columns = [
        "ECGI", "KMEANS_CLUSTER", "KMEANS_DISTANCE_TO_CENTROID",
        "KMEANS_SELECTED_K", "KMEANS_SELECTION_STATUS", "KMEANS_SILHOUETTE",
        "IF_DOMINANT_MAX_PERCENTILE", "IF_DOMINANT_SELECTED_FEATURE_COUNT",
        "IF_DOMINANT_ECGI_DETECTED_FLAG", "MODE2_ECGI_RETAINED_FLAG",
        "MODE2_INDEPENDENT_ONE_CHANGE_EXTREME_FLAG",
    ]
    if bool(args.kmeans_rule_compatibility):
        columns += [
            "KMEANS_MAJORITY_RULE_CLUSTER", "KMEANS_CLUSTER_SIZE",
            "KMEANS_RULE_PURITY", "KMEANS_RULE_COMPATIBILITY",
        ]
    available = [column for column in columns if column in data.columns]
    if "ECGI" not in available:
        return pd.DataFrame(columns=columns)
    return (
        data.loc[
            col_num(data, "IF_DOMINANT_ECGI_DETECTED_FLAG", 0).eq(1),
            available,
        ]
        .sort_values(["ECGI"])
        .groupby("ECGI", as_index=False).head(1)
    )


def _enrich_mode2_summary(
    summary: pd.DataFrame,
    data: pd.DataFrame,
    args: argparse.Namespace,
) -> pd.DataFrame:
    if summary is None or summary.empty:
        return summary
    result = summary.copy()
    result["DETECTION_MODE_TAG"] = MODE2_LABEL
    result["MODE2_IF_DOMINANT_DETECTED_FLAG"] = 1
    audit = _ecgi_kmeans_audit(data, args)
    if not audit.empty:
        result = result.merge(audit, on="ECGI", how="left", validate="one_to_one")
    return result


def _build_comparison_summary(
    data: pd.DataFrame,
    args: argparse.Namespace,
) -> pd.DataFrame:
    if data is None or data.empty:
        return pd.DataFrame()
    mode1_data = _restore_mode1_view(data)
    mode2_data = _build_mode2_view(data, args)
    mode1 = _build_signature_dominant_summary(mode1_data, args)
    mode2 = _enrich_mode2_summary(
        _build_signature_dominant_summary(mode2_data, args),
        mode2_data,
        args,
    )
    if mode1.empty and mode2.empty:
        return pd.DataFrame()
    mode1_index = mode1.set_index("ECGI", drop=False) if not mode1.empty else pd.DataFrame()
    mode2_index = mode2.set_index("ECGI", drop=False) if not mode2.empty else pd.DataFrame()
    ecgis = sorted(
        set(mode1.get("ECGI", pd.Series(dtype=str)).astype(str))
        | set(mode2.get("ECGI", pd.Series(dtype=str)).astype(str))
    )
    rows: List[Dict[str, Any]] = []
    for ecgi in ecgis:
        has_mode1 = not mode1.empty and ecgi in mode1_index.index
        has_mode2 = not mode2.empty and ecgi in mode2_index.index
        source = mode1_index.loc[ecgi] if has_mode1 else mode2_index.loc[ecgi]
        if isinstance(source, pd.DataFrame):
            source = source.iloc[0]
        row = source.to_dict()
        mode1_cluster = (
            str(mode1_index.loc[ecgi, "PRIMARY_ANOMALY_CLUSTER"])
            if has_mode1 else ""
        )
        mode2_cluster = (
            str(mode2_index.loc[ecgi, "PRIMARY_ANOMALY_CLUSTER"])
            if has_mode2 else ""
        )
        mode1_score = (
            scalar_float(mode1_index.loc[ecgi, "ECGI_ANOMALY_RISK_SCORE"])
            if has_mode1 else np.nan
        )
        mode2_score = (
            scalar_float(mode2_index.loc[ecgi, "ECGI_ANOMALY_RISK_SCORE"])
            if has_mode2 else np.nan
        )
        mode1_all = (
            str(mode1_index.loc[ecgi, "ALL_MATCHED_CLUSTERS"])
            if has_mode1 and "ALL_MATCHED_CLUSTERS" in mode1_index.columns else ""
        )
        mode2_all = (
            str(mode2_index.loc[ecgi, "ALL_MATCHED_CLUSTERS"])
            if has_mode2 and "ALL_MATCHED_CLUSTERS" in mode2_index.columns else ""
        )
        row.update({
            "ECGI": ecgi,
            "MODE1_SIGNATURE_DOMINANT_DETECTED_FLAG": int(has_mode1),
            "MODE2_IF_DOMINANT_DETECTED_FLAG": int(has_mode2),
            "MODE1_PRIMARY_ANOMALY_CLUSTER": mode1_cluster,
            "MODE2_PRIMARY_ANOMALY_CLUSTER": mode2_cluster,
            "MODE1_ANOMALY_RISK_SCORE": mode1_score,
            "MODE2_ANOMALY_RISK_SCORE": mode2_score,
            "MODE1_ALL_MATCHED_CLUSTERS": mode1_all,
            "MODE2_ALL_MATCHED_CLUSTERS": mode2_all,
            "DETECTOR_AGREEMENT": (
                "AGREE_IF_AND_SIGNATURE" if has_mode1 and has_mode2
                else "JUST_SIGNATURE" if has_mode1
                else "JUST_IF"
            ),
            "CLUSTER_AGREEMENT": (
                "SAME_CLUSTER" if has_mode1 and has_mode2
                and mode1_cluster == mode2_cluster
                else "DIFFERENT_CLUSTER" if has_mode1 and has_mode2
                else "NOT_COMPARABLE"
            ),
            "DETECTION_MODE_TAG": (
                f"{MODE1_LABEL}; {MODE2_LABEL}" if has_mode1 and has_mode2
                else MODE1_LABEL if has_mode1 else MODE2_LABEL
            ),
            "PRIMARY_ANOMALY_CLUSTER": mode1_cluster or mode2_cluster,
            "ECGI_ANOMALY_RISK_SCORE": (
                mode1_score if has_mode1 else mode2_score
            ),
        })
        if has_mode2:
            for column in [
                "KMEANS_CLUSTER", "KMEANS_DISTANCE_TO_CENTROID",
                "KMEANS_SELECTED_K", "KMEANS_SELECTION_STATUS",
                "KMEANS_SILHOUETTE",
                "KMEANS_MAJORITY_RULE_CLUSTER", "KMEANS_CLUSTER_SIZE",
                "KMEANS_RULE_PURITY", "KMEANS_RULE_COMPATIBILITY",
            ]:
                if column in mode2_index.columns:
                    row[column] = mode2_index.loc[ecgi, column]
        rows.append(row)
    result = pd.DataFrame(rows)
    result = result.sort_values(
        ["ECGI_ANOMALY_RISK_SCORE", "ECGI"],
        ascending=[False, True],
    ).reset_index(drop=True)
    if "INCIDENT_RANK" in result.columns:
        result.drop(columns=["INCIDENT_RANK"], inplace=True)
    result.insert(0, "INCIDENT_RANK", np.arange(1, len(result) + 1))
    if result["ECGI"].duplicated().any():
        raise AssertionError("Comparison union created duplicate ECGI rows.")
    return result


def build_ecgi_summary(data: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    """Select the requested Mode 1, Mode 2, or comparison output contract."""
    if args.detection_mode == "signature_dominant":
        summary = _build_signature_dominant_summary(data, args)
        if not summary.empty:
            summary["DETECTION_MODE_TAG"] = MODE1_LABEL
            summary["MODE1_SIGNATURE_DOMINANT_DETECTED_FLAG"] = 1
        return summary
    if args.detection_mode == "if_dominant":
        return _enrich_mode2_summary(
            _build_signature_dominant_summary(data, args), data, args
        )
    return _build_comparison_summary(data, args)


def build_change_path_summary(events: pd.DataFrame) -> pd.DataFrame:
    if events is None or events.empty:
        return pd.DataFrame(columns=["POPULATION", "ECGI", "FEATURE"])
    data = events.copy()
    data["LOAD_DATE"] = pd.to_datetime(data["LOAD_DATE"], errors="coerce").dt.normalize()
    data = data.sort_values(["POPULATION", "ECGI", "FEATURE", "LOAD_DATE"])
    data["TRANSITION_TEXT"] = data.apply(
        lambda row: (
            f"{row['LOAD_DATE'].date()}: {row.get('PREVIOUS_SOURCE_VALUE','')}"
            f" -> {row.get('CURRENT_SOURCE_VALUE','')}"
        ) if pd.notna(row["LOAD_DATE"]) else "",
        axis=1,
    )
    return data.groupby(["POPULATION", "ECGI", "FEATURE"], as_index=False).agg(
        CHANGE_DATES=(
            "LOAD_DATE",
            lambda values: ", ".join(
                pd.to_datetime(values, errors="coerce").dropna().drop_duplicates()
                .sort_values().dt.strftime("%Y-%m-%d").tolist()
            ),
        ),
        SOURCE_TRANSITIONS_BY_DATE=(
            "TRANSITION_TEXT", lambda values: "; ".join(item for item in values if item)
        ),
    )


def csv_write(
    frame: Optional[pd.DataFrame],
    path: Path,
    empty_columns: Sequence[str] = (),
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    output = pd.DataFrame() if frame is None else frame
    if output.empty and len(output.columns) == 0:
        output = pd.DataFrame(columns=list(empty_columns))
    output.to_csv(path, index=False)


def cluster_description(cluster: str) -> str:
    return {
        "HIGH_FREQUENCY_CHANGES": "One monitored source feature reached the configured recurrent-change-date threshold.",
        "REAL_BACK_AND_FORTH": "A monitored source feature followed a real reverse transition such as A->B->A.",
        "MULTI_FEATURE_CHANGES": "The configured number of monitored features changed across different dates without one date dominating most features.",
        "RARE_CHANGES": (
            "Unusual recurrent behavior, an objective current/repeated source conflict, "
            "or a conservatively calibrated ultra-tail one-change path versus same-age "
            "held-out normal one-change history."
        ),
    }.get(str(cluster), str(cluster))


def write_markdown_report(
    path: Path,
    summary: pd.DataFrame,
    run_info: RunInfo,
    args: argparse.Namespace,
) -> None:
    lines = [
        "# CESMLC V9.8.5 Three-Mode Window-Mismatch Anomaly Report",
        "",
        f"- Script version: `{SCRIPT_VERSION}`",
        f"- Detection mode: **{args.detection_mode}**",
        f"- IF-dominant percentile: **{float(args.if_dominant_percentile):.4f}**",
        f"- Diagnostic KMeans: **{'auto' if args.kmeans_auto else int(args.kmeans_clusters)}**",
        f"- Analysis window: **{run_info.window_start.date()} to {run_info.window_end.date()}**",
        f"- Candidate gate: **{args.candidate_gate}**",
        f"- Candidate gate lookback: **{int(args.candidate_gate_days or args.days)} dates**",
        f"- Minimum mismatch dates for gate: **{int(args.candidate_mismatch_min_days)}**",
        f"- Trailing recency-context window: **{run_info.stable_start.date()} to {run_info.window_end.date()}**",
        f"- `{args.candidate_gate}` candidate ECGIs: **{run_info.candidate_count:,}**",
        f"- Real normal-baseline ECGIs: **{run_info.baseline_count:,}**",
        f"- Final incidents: **{len(summary):,}**",
        "",
        "`SITE_TYPE` and `IS_COW_SITE` are attached after detection as review context only. Missing matches "
        "remain in processing as `UNKNOWN`; COW is never added to training groups or suppression. "
        "Each feature model is a pooled multi-seed Isolation Forest calibrated against untouched normal history. "
        "`PHYSICALCELLID` is absent from history engineering and all decision paths. No monitored source "
        f"change during the last {int(args.stable_days)} days is recency context only and cannot erase a "
        "qualifying pattern from the full analysis window. "
        "A rule-qualified incident is never deleted by a second score threshold.",
        "",
        "## Final cluster definitions",
        "",
    ]
    for cluster in FINAL_CLUSTERS:
        lines.append(f"- `{cluster}`: {cluster_description(cluster)}")
    lines += ["", "## Incident counts", ""]
    if summary.empty:
        lines.append("No ECGI qualified for the four requested clusters.")
    else:
        lines += [
            "| Cluster | ECGIs |",
            "|---|---:|",
        ]
        for cluster in FINAL_CLUSTERS:
            count = int(summary["PRIMARY_ANOMALY_CLUSTER"].eq(cluster).sum())
            lines.append(f"| {cluster} | {count:,} |")
        lines += [
            "",
            "## Top ECGI incidents",
            "",
            "| Rank | ECGI | SITE_TYPE | COW | Cluster | Primary evidence feature | Risk | Change days | Recent change days | All matched clusters |",
            "|---:|---|---|---:|---|---|---:|---:|---:|---|",
        ]
        for row in summary.head(int(args.final_report_top_n)).itertuples(index=False):
            lines.append(
                f"| {scalar_int(row.INCIDENT_RANK)} | {row.ECGI_LABEL} | {row.SITE_TYPE} | "
                f"{scalar_int(row.IS_COW_SITE)} | {row.PRIMARY_ANOMALY_CLUSTER} | "
                f"{row.PRIMARY_ABNORMAL_FEATURE} | {scalar_float(row.ECGI_ANOMALY_RISK_SCORE):.1f} | "
                f"{scalar_int(row.PRIMARY_FEATURE_CHANGE_DAYS)} | "
                f"{scalar_int(row.PRIMARY_FEATURE_RECENT_CHANGE_DAYS)} | "
                f"{row.ALL_MATCHED_CLUSTERS} |"
            )
    path.write_text("\n".join(lines), encoding="utf-8")


def build_decision_funnel(
    data: pd.DataFrame,
    summary: pd.DataFrame,
    run_info: RunInfo,
) -> pd.DataFrame:
    candidate = data[data["POPULATION"].eq("CANDIDATE")].copy() if not data.empty else pd.DataFrame()

    def record(stage: str, mask: Optional[pd.Series] = None, ecgi_level: bool = False) -> Dict[str, Any]:
        if candidate.empty:
            return {"STAGE": stage, "ROW_COUNT": 0, "ECGI_COUNT": 0}
        subset = candidate if mask is None else candidate.loc[pd.Series(mask, index=candidate.index).fillna(False)]
        ecgis = int(subset["ECGI"].nunique()) if not subset.empty else 0
        return {
            "STAGE": stage,
            "ROW_COUNT": ecgis if ecgi_level else int(len(subset)),
            "ECGI_COUNT": ecgis,
        }

    if candidate.empty:
        stages = [{
            "STAGE": "GATED_CANDIDATE_ECGIS",
            "ROW_COUNT": int(run_info.candidate_count),
            "ECGI_COUNT": int(run_info.candidate_count),
        }]
        stages.append({"STAGE": "FINAL_REPORTABLE_ECGI_INCIDENTS", "ROW_COUNT": 0, "ECGI_COUNT": 0})
        return pd.DataFrame(stages)
    changed = col_num(candidate, "SOURCE_CHANGE_DAYS", 0).gt(0)
    quiet_column = (
        "QUIET_LAST_N_DAYS_FLAG"
        if "QUIET_LAST_N_DAYS_FLAG" in candidate.columns
        else "STABLE_LAST_N_DAYS_FLAG"
    )
    quiet = col_num(candidate, quiet_column, 0).eq(1)
    recurrent = col_num(candidate, "SOURCE_CHANGE_DAYS", 0).ge(2)
    raw_feature = col_num(candidate, "RAW_QUALIFIED_FEATURE_FLAG", 0).eq(1)
    raw_multi = col_num(candidate, "RAW_MULTI_FEATURE_TOPOLOGY_FLAG", 0).eq(1)
    reportable = col_num(candidate, "REPORTABLE_EVIDENCE_ROW_FLAG", 0).eq(1)
    stages = [
        {
            "STAGE": "GATED_CANDIDATE_ECGIS",
            "ROW_COUNT": int(run_info.candidate_count),
            "ECGI_COUNT": int(run_info.candidate_count),
        },
        record("CANDIDATE_MONITORED_FEATURE_ROWS"),
        {
            "STAGE": "ENGINEERED_NONPCI_CANDIDATE_ECGIS",
            "ROW_COUNT": int(candidate["ECGI"].nunique()),
            "ECGI_COUNT": int(candidate["ECGI"].nunique()),
        },
        {
            "STAGE": "CANDIDATE_ECGIS_WITHOUT_USABLE_NONPCI_HISTORY",
            "ROW_COUNT": max(0, int(run_info.candidate_count) - int(candidate["ECGI"].nunique())),
            "ECGI_COUNT": max(0, int(run_info.candidate_count) - int(candidate["ECGI"].nunique())),
        },
        record("ENGINEERED_PHYSICALCELLID_ROWS", candidate["FEATURE"].eq("PHYSICALCELLID")),
        record("BOUNDARY_CHANGE_EVENTS_RECOVERED", col_num(candidate, "BOUNDARY_CHANGE_EVENTS_RECOVERED", 0).gt(0)),
        record("CHANGED_MONITORED_FEATURE_ROWS", changed),
        record("ONE_CHANGE_DATE_FEATURE_ROWS", col_num(candidate, "SOURCE_CHANGE_DAYS", 0).eq(1)),
        record("RECURRENT_FEATURE_ROWS_ALL_CANDIDATES", recurrent),
        record("QUIET_LAST_N_DAYS_CONTEXT_ECGIS", quiet, ecgi_level=True),
        record("RECENT_CHANGE_CONTEXT_ECGIS", ~quiet, ecgi_level=True),
        record("RAW_QUALIFIED_FEATURE_ROWS", raw_feature),
        record("RAW_QUALIFIED_QUIET_FEATURE_ROWS", raw_feature & quiet),
        record(
            "OLD_STABILITY_POLICY_WOULD_DROP_ECGIS",
            col_num(candidate, "WOULD_OLD_STABILITY_OVERRIDE_DROP_FLAG", 0).eq(1),
            ecgi_level=True,
        ),
        record("RAW_HIGH_FREQUENCY_FEATURE_ROWS", col_num(candidate, "RAW_HIGH_FREQUENCY_CHANGES_FLAG", 0).eq(1)),
        record("RAW_BACKFORTH_HIGH_FREQUENCY_OVERLAP_ROWS", col_num(candidate, "RAW_BACKFORTH_AND_HIGH_FREQUENCY_OVERLAP_FLAG", 0).eq(1)),
        record("FINAL_HIGH_FREQUENCY_FEATURE_ROWS", col_num(candidate, "HIGH_FREQUENCY_CHANGES_FLAG", 0).eq(1)),
        record("FINAL_BACK_AND_FORTH_FEATURE_ROWS", col_num(candidate, "REAL_BACK_AND_FORTH_FLAG", 0).eq(1)),
        record("RAW_MULTI_FEATURE_TOPOLOGY_ECGIS", col_num(candidate, "RAW_MULTI_FEATURE_TOPOLOGY_FLAG", 0).eq(1), ecgi_level=True),
        record("FINAL_MULTI_FEATURE_ECGIS", col_num(candidate, "MULTI_FEATURE_CHANGES_FLAG", 0).eq(1), ecgi_level=True),
        record("RAW_RECURRENT_MODEL_RARE_ROWS", col_num(candidate, "RAW_RARE_RECURRENT_MODEL_FLAG", 0).eq(1)),
        record("RAW_ONE_CHANGE_CONDITIONAL_MODEL_DISCOVERY_ROWS", col_num(candidate, "RAW_RARE_ONE_CHANGE_CONDITIONAL_MODEL_FLAG", 0).eq(1)),
        record("RAW_CURRENT_OR_RECURRENT_CONFLICT_ROWS", col_num(candidate, "RAW_RARE_CONFLICT_FLAG", 0).eq(1)),
        record("FINAL_RARE_FEATURE_ROWS", col_num(candidate, "RARE_CHANGES_FLAG", 0).eq(1)),
        record("ONE_DAY_EXTREME_RARE_ROWS_REJECTED", col_num(candidate, "ONE_DAY_RARE_GUARD_FLAG", 0).eq(1)),
        record("INDIVIDUALLY_REPORTABLE_FEATURE_ROWS", col_num(candidate, "FEATURE_REPORTABLE_FLAG", 0).eq(1)),
        record("MULTI_FEATURE_CONTRIBUTOR_ROWS", col_num(candidate, "MULTI_FEATURE_CONTRIBUTOR_FLAG", 0).eq(1)),
        record("REPORTABLE_EVIDENCE_ROWS", col_num(candidate, "REPORTABLE_EVIDENCE_ROW_FLAG", 0).eq(1)),
        record("REPORTABLE_QUIET_ECGIS", reportable & quiet, ecgi_level=True),
        record(
            "DROPPED_BY_RECENCY",
            col_num(candidate, "DROPPED_BY_RECENCY_FLAG", 0).eq(1),
            ecgi_level=True,
        ),
        {
            "STAGE": "FINAL_REPORTABLE_ECGI_INCIDENTS",
            "ROW_COUNT": int(len(summary)),
            "ECGI_COUNT": int(len(summary)),
        },
    ]
    return pd.DataFrame(stages)


def build_feature_detection_diagnostics(
    data: pd.DataFrame,
    args: argparse.Namespace,
) -> pd.DataFrame:
    """Expose where evidence exists and where each decision branch retains it."""
    if data is None or data.empty:
        return pd.DataFrame()
    candidate = data[data["POPULATION"].eq("CANDIDATE")].copy()
    if candidate.empty:
        return pd.DataFrame()
    quiet_column = (
        "QUIET_LAST_N_DAYS_FLAG"
        if "QUIET_LAST_N_DAYS_FLAG" in candidate.columns
        else "STABLE_LAST_N_DAYS_FLAG"
    )
    candidate["_OBSERVED"] = col_num(candidate, "OBSERVED_DAYS", 0).gt(0).astype(int)
    candidate["_CHANGED"] = col_num(candidate, "SOURCE_CHANGE_DAYS", 0).gt(0).astype(int)
    candidate["_ONE_DATE"] = col_num(candidate, "SOURCE_CHANGE_DAYS", 0).eq(1).astype(int)
    candidate["_RECURRENT"] = col_num(candidate, "SOURCE_CHANGE_DAYS", 0).ge(2).astype(int)
    candidate["_FOUR_PLUS"] = col_num(candidate, "SOURCE_CHANGE_DAYS", 0).ge(
        int(args.high_frequency_min_change_days)
    ).astype(int)
    candidate["_MODEL_SCORED"] = col_num(candidate, "MODEL_AVAILABLE_FLAG", 0).eq(1).astype(int)
    candidate["_MODEL_TAIL"] = col_num(candidate, "MODEL_NORMAL_PERCENTILE", 0).ge(
        float(args.rare_model_percentile)
    ).astype(int)
    candidate["_BOUNDARY_RECOVERED"] = col_num(
        candidate, "BOUNDARY_CHANGE_EVENTS_RECOVERED", 0
    ).gt(0).astype(int)
    candidate["_ONE_CHANGE_REFERENCE_AVAILABLE"] = col_num(
        candidate, "MODEL_ONE_CHANGE_CONDITIONAL_AVAILABLE_FLAG", 0
    ).eq(1).astype(int)
    candidate["_QUIET"] = col_num(candidate, quiet_column, 0).eq(1).astype(int)
    candidate["_RAW_QUALIFIED"] = col_num(
        candidate, "RAW_QUALIFIED_FEATURE_FLAG", 0
    ).eq(1).astype(int)
    candidate["_RAW_QUALIFIED_QUIET"] = (
        candidate["_RAW_QUALIFIED"].eq(1) & candidate["_QUIET"].eq(1)
    ).astype(int)
    candidate["_OLD_OVERRIDE_LOSS"] = col_num(
        candidate, "WOULD_OLD_STABILITY_OVERRIDE_DROP_FLAG", 0
    ).eq(1).astype(int)
    diagnostic = candidate.groupby("FEATURE", as_index=False).agg(
        CANDIDATE_FEATURE_ROWS=("ECGI", "size"),
        CANDIDATE_ECGIS=("ECGI", "nunique"),
        OBSERVED_HISTORY_ROWS=("_OBSERVED", "sum"),
        CHANGED_FEATURE_ROWS=("_CHANGED", "sum"),
        TOTAL_CHANGE_EVENTS=("SOURCE_CHANGE_EVENTS", "sum"),
        TOTAL_CHANGE_DAYS=("SOURCE_CHANGE_DAYS", "sum"),
        BOUNDARY_RECOVERED_FEATURE_ROWS=("_BOUNDARY_RECOVERED", "sum"),
        BOUNDARY_RECOVERED_CHANGE_EVENTS=("BOUNDARY_CHANGE_EVENTS_RECOVERED", "sum"),
        ONE_CHANGE_DATE_ROWS=("_ONE_DATE", "sum"),
        RECURRENT_ROWS=("_RECURRENT", "sum"),
        FOUR_PLUS_CHANGE_DATE_ROWS=("_FOUR_PLUS", "sum"),
        RAW_BACK_AND_FORTH_ROWS=("RAW_REAL_BACK_AND_FORTH_FLAG", "sum"),
        RAW_HIGH_FREQUENCY_ROWS=("RAW_HIGH_FREQUENCY_CHANGES_FLAG", "sum"),
        RAW_BACKFORTH_HIGH_FREQUENCY_OVERLAP_ROWS=("RAW_BACKFORTH_AND_HIGH_FREQUENCY_OVERLAP_FLAG", "sum"),
        RAW_RECURRENT_MODEL_RARE_ROWS=("RAW_RARE_RECURRENT_MODEL_FLAG", "sum"),
        ONE_CHANGE_CONDITIONAL_REFERENCE_AVAILABLE_ROWS=("_ONE_CHANGE_REFERENCE_AVAILABLE", "sum"),
        RAW_ONE_CHANGE_CONDITIONAL_MODEL_DISCOVERY_ROWS=("RAW_RARE_ONE_CHANGE_CONDITIONAL_MODEL_FLAG", "sum"),
        RAW_CURRENT_OR_RECURRENT_CONFLICT_ROWS=("RAW_RARE_CONFLICT_FLAG", "sum"),
        RAW_QUALIFIED_ROWS=("_RAW_QUALIFIED", "sum"),
        MODEL_SCORED_ROWS=("_MODEL_SCORED", "sum"),
        MODEL_TAIL_ROWS=("_MODEL_TAIL", "sum"),
        QUIET_CONTEXT_ROWS=("_QUIET", "sum"),
        RAW_QUALIFIED_QUIET_ROWS=("_RAW_QUALIFIED_QUIET", "sum"),
        FINAL_FEATURE_REPORTABLE_ROWS=("FEATURE_REPORTABLE_FLAG", "sum"),
        MULTI_CONTRIBUTOR_ROWS=("MULTI_FEATURE_CONTRIBUTOR_FLAG", "sum"),
        FINAL_REPORTABLE_EVIDENCE_ROWS=("REPORTABLE_EVIDENCE_ROW_FLAG", "sum"),
        OLD_STABILITY_POLICY_WOULD_DROP_ROWS=("_OLD_OVERRIDE_LOSS", "sum"),
        DROPPED_BY_RECENCY_ROWS=("DROPPED_BY_RECENCY_FLAG", "sum"),
    )
    return diagnostic.sort_values(
        ["FINAL_REPORTABLE_EVIDENCE_ROWS", "RAW_QUALIFIED_ROWS", "FEATURE"],
        ascending=[False, False, True],
    ).reset_index(drop=True)


def build_daily_representative_reconciliation(
    data: pd.DataFrame,
    run_info: RunInfo,
    args: argparse.Namespace,
) -> pd.DataFrame:
    """Summarize modal-versus-V9.6-lexical-MIN history without changing decisions.

    The comparison deliberately holds the V9.8 candidate population, normalized
    values, analysis window, and boundary anchor constant.  It therefore isolates
    only the daily representative policy.  It is not an exact V9.6 replay: V9.6
    also included PHYSICALCELLID and defaulted coordinates to five decimals.
    """
    columns = [
        "FEATURE", "GATED_CANDIDATE_ECGIS",
        "CANDIDATE_ECGIS_WITH_NON_NULL_FEATURE_HISTORY",
        "CANDIDATE_ECGIS_WITHOUT_NON_NULL_FEATURE_HISTORY",
        "MODAL_CHANGED_ECGIS", "LEGACY_MIN_CHANGED_ECGIS",
        "BOTH_CHANGED_ECGIS", "LEGACY_MIN_ONLY_CHANGED_ECGIS",
        "MODAL_ONLY_CHANGED_ECGIS", "EITHER_CHANGED_ECGIS",
        "MODAL_CHANGE_EVENTS", "MODAL_CHANGE_DAYS",
        "LEGACY_MIN_CHANGE_EVENTS", "LEGACY_MIN_CHANGE_DAYS",
        "LEGACY_MIN_ONLY_CHANGE_DAYS", "MODAL_ONLY_CHANGE_DAYS",
        "MODAL_AND_LEGACY_CHANGE_DAYS",
        "DAILY_REPRESENTATIVE_DISAGREEMENT_ECGIS",
        "DAILY_REPRESENTATIVE_DISAGREEMENT_DAYS",
        "SAME_DAY_CONFLICT_ECGIS", "SAME_DAY_CONFLICT_DAYS",
        "LEGACY_MIN_ONLY_WITH_CONFLICT_ECGIS",
        "LEGACY_MIN_ONLY_WITHOUT_CONFLICT_ECGIS",
        "BOUNDARY_RECOVERED_MODAL_CHANGE_EVENTS",
        "LEGACY_MINUS_MODAL_CHANGED_ECGIS",
        "LEGACY_MINUS_MODAL_CHANGE_DAYS", "CHANGED_ECGI_OVERLAP_JACCARD",
        "LATLON_DECIMALS_USED", "COMPARISON_SCOPE",
        "DECISION_REPRESENTATIVE", "LEGACY_REPRESENTATIVE_ROLE",
        "PHYSICALCELLID_INCLUDED_FLAG", "SITE_TYPE_USED_IN_COMPARISON_FLAG",
    ]
    if data is None or data.empty:
        return pd.DataFrame(columns=columns)
    candidate = data.loc[data["POPULATION"].astype(str).eq("CANDIDATE")].copy()
    if candidate.empty:
        return pd.DataFrame(columns=columns)
    if candidate["FEATURE"].astype(str).str.upper().eq("PHYSICALCELLID").any():
        raise AssertionError(
            "PHYSICALCELLID reached the modal-versus-legacy reconciliation path."
        )

    candidate["_MODAL_CHANGED"] = col_num(
        candidate, "SOURCE_CHANGE_DAYS", 0
    ).gt(0).astype(int)
    candidate["_LEGACY_CHANGED"] = col_num(
        candidate, "LEGACY_MIN_CHANGE_DAYS", 0
    ).gt(0).astype(int)
    candidate["_BOTH_CHANGED"] = (
        candidate["_MODAL_CHANGED"].eq(1)
        & candidate["_LEGACY_CHANGED"].eq(1)
    ).astype(int)
    candidate["_LEGACY_ONLY_CHANGED"] = (
        candidate["_MODAL_CHANGED"].eq(0)
        & candidate["_LEGACY_CHANGED"].eq(1)
    ).astype(int)
    candidate["_MODAL_ONLY_CHANGED"] = (
        candidate["_MODAL_CHANGED"].eq(1)
        & candidate["_LEGACY_CHANGED"].eq(0)
    ).astype(int)
    candidate["_EITHER_CHANGED"] = (
        candidate["_MODAL_CHANGED"].eq(1)
        | candidate["_LEGACY_CHANGED"].eq(1)
    ).astype(int)
    candidate["_REPRESENTATIVE_DISAGREEMENT"] = col_num(
        candidate, "DAILY_REPRESENTATIVE_DISAGREEMENT_DAYS", 0
    ).gt(0).astype(int)
    candidate["_SAME_DAY_CONFLICT"] = col_num(
        candidate, "SAME_DAY_CONFLICT_DAYS", 0
    ).gt(0).astype(int)
    candidate["_LEGACY_ONLY_WITH_CONFLICT"] = (
        candidate["_LEGACY_ONLY_CHANGED"].eq(1)
        & candidate["_SAME_DAY_CONFLICT"].eq(1)
    ).astype(int)
    candidate["_LEGACY_ONLY_WITHOUT_CONFLICT"] = (
        candidate["_LEGACY_ONLY_CHANGED"].eq(1)
        & candidate["_SAME_DAY_CONFLICT"].eq(0)
    ).astype(int)

    reconciliation = candidate.groupby("FEATURE", as_index=False).agg(
        CANDIDATE_ECGIS_WITH_NON_NULL_FEATURE_HISTORY=("ECGI", "nunique"),
        MODAL_CHANGED_ECGIS=("_MODAL_CHANGED", "sum"),
        LEGACY_MIN_CHANGED_ECGIS=("_LEGACY_CHANGED", "sum"),
        BOTH_CHANGED_ECGIS=("_BOTH_CHANGED", "sum"),
        LEGACY_MIN_ONLY_CHANGED_ECGIS=("_LEGACY_ONLY_CHANGED", "sum"),
        MODAL_ONLY_CHANGED_ECGIS=("_MODAL_ONLY_CHANGED", "sum"),
        EITHER_CHANGED_ECGIS=("_EITHER_CHANGED", "sum"),
        MODAL_CHANGE_EVENTS=("SOURCE_CHANGE_EVENTS", "sum"),
        MODAL_CHANGE_DAYS=("SOURCE_CHANGE_DAYS", "sum"),
        LEGACY_MIN_CHANGE_EVENTS=("LEGACY_MIN_CHANGE_EVENTS", "sum"),
        LEGACY_MIN_CHANGE_DAYS=("LEGACY_MIN_CHANGE_DAYS", "sum"),
        LEGACY_MIN_ONLY_CHANGE_DAYS=("LEGACY_MIN_ONLY_CHANGE_DAYS", "sum"),
        MODAL_ONLY_CHANGE_DAYS=("MODAL_ONLY_CHANGE_DAYS", "sum"),
        MODAL_AND_LEGACY_CHANGE_DAYS=("MODAL_AND_LEGACY_CHANGE_DAYS", "sum"),
        DAILY_REPRESENTATIVE_DISAGREEMENT_ECGIS=(
            "_REPRESENTATIVE_DISAGREEMENT", "sum"
        ),
        DAILY_REPRESENTATIVE_DISAGREEMENT_DAYS=(
            "DAILY_REPRESENTATIVE_DISAGREEMENT_DAYS", "sum"
        ),
        SAME_DAY_CONFLICT_ECGIS=("_SAME_DAY_CONFLICT", "sum"),
        SAME_DAY_CONFLICT_DAYS=("SAME_DAY_CONFLICT_DAYS", "sum"),
        LEGACY_MIN_ONLY_WITH_CONFLICT_ECGIS=(
            "_LEGACY_ONLY_WITH_CONFLICT", "sum"
        ),
        LEGACY_MIN_ONLY_WITHOUT_CONFLICT_ECGIS=(
            "_LEGACY_ONLY_WITHOUT_CONFLICT", "sum"
        ),
        BOUNDARY_RECOVERED_MODAL_CHANGE_EVENTS=(
            "BOUNDARY_CHANGE_EVENTS_RECOVERED", "sum"
        ),
    )
    reconciliation.insert(
        1, "GATED_CANDIDATE_ECGIS", int(run_info.candidate_count)
    )
    reconciliation.insert(
        3,
        "CANDIDATE_ECGIS_WITHOUT_NON_NULL_FEATURE_HISTORY",
        (
            int(run_info.candidate_count)
            - col_num(
                reconciliation,
                "CANDIDATE_ECGIS_WITH_NON_NULL_FEATURE_HISTORY",
                0,
            )
        ).clip(lower=0).astype(int),
    )
    reconciliation["LEGACY_MINUS_MODAL_CHANGED_ECGIS"] = (
        col_num(reconciliation, "LEGACY_MIN_CHANGED_ECGIS", 0)
        - col_num(reconciliation, "MODAL_CHANGED_ECGIS", 0)
    )
    reconciliation["LEGACY_MINUS_MODAL_CHANGE_DAYS"] = (
        col_num(reconciliation, "LEGACY_MIN_CHANGE_DAYS", 0)
        - col_num(reconciliation, "MODAL_CHANGE_DAYS", 0)
    )
    reconciliation["CHANGED_ECGI_OVERLAP_JACCARD"] = (
        col_num(reconciliation, "BOTH_CHANGED_ECGIS", 0)
        / col_num(reconciliation, "EITHER_CHANGED_ECGIS", 0).replace(0, np.nan)
    ).fillna(1.0)
    reconciliation["LATLON_DECIMALS_USED"] = int(args.latlon_decimals)
    reconciliation["COMPARISON_SCOPE"] = (
        "SAME_V98_NORMALIZATION_POPULATION_WINDOW_AND_BOUNDARY_ANCHOR;"
        "DAILY_REPRESENTATIVE_ONLY"
    )
    reconciliation["DECISION_REPRESENTATIVE"] = "DAILY_MODAL_THEN_LEXICAL_TIE_BREAK"
    reconciliation["LEGACY_REPRESENTATIVE_ROLE"] = (
        "LEXICAL_MIN_DIAGNOSTIC_ONLY_NEVER_USED_FOR_DECISIONS"
    )
    reconciliation["PHYSICALCELLID_INCLUDED_FLAG"] = 0
    reconciliation["SITE_TYPE_USED_IN_COMPARISON_FLAG"] = 0
    return reconciliation[columns].sort_values(
        ["LEGACY_MIN_ONLY_CHANGED_ECGIS", "LEGACY_MINUS_MODAL_CHANGE_DAYS", "FEATURE"],
        ascending=[False, False, True],
    ).reset_index(drop=True)


def write_mode_comparison_artifacts(
    out_dir: Path,
    summary: pd.DataFrame,
    args: argparse.Namespace,
) -> None:
    """Write the requested comparison table and two optional pie charts."""
    if args.detection_mode != "compare":
        return
    csv_write(
        summary,
        out_dir / "cesmlc_v985_mode_comparison.csv",
        [
            "INCIDENT_RANK", "ECGI",
            "MODE1_SIGNATURE_DOMINANT_DETECTED_FLAG",
        "MODE2_IF_DOMINANT_DETECTED_FLAG",
            "MODE1_PRIMARY_ANOMALY_CLUSTER",
            "MODE2_PRIMARY_ANOMALY_CLUSTER",
            "MODE1_ANOMALY_RISK_SCORE", "MODE2_ANOMALY_RISK_SCORE",
            "MODE1_ALL_MATCHED_CLUSTERS", "MODE2_ALL_MATCHED_CLUSTERS",
            "DETECTOR_AGREEMENT", "CLUSTER_AGREEMENT",
            "DETECTION_MODE_TAG",
        ],
    )
    if bool(args.no_comparison_charts) or summary.empty:
        return
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        log(
            f"WARNING: comparison CSV written, but pie charts were skipped "
            f"because matplotlib is unavailable ({exc}).",
            1,
        )
        return

    def pie(series: pd.Series, title: str, filename: str) -> None:
        counts = series.fillna("UNKNOWN").astype(str).value_counts()
        figure, axis = plt.subplots(figsize=(8, 6))
        axis.pie(
            counts.to_numpy(),
            labels=counts.index.tolist(),
            autopct=lambda value: f"{value:.1f}%",
            startangle=90,
        )
        axis.set_title(title)
        axis.axis("equal")
        figure.tight_layout()
        figure.savefig(out_dir / filename, dpi=160, bbox_inches="tight")
        plt.close(figure)

    figure, axes = plt.subplots(1, 2, figsize=(14, 6))
    for axis, column, title in [
        (
            axes[0],
            "MODE1_PRIMARY_ANOMALY_CLUSTER",
            "Mode 1: IF + Signature",
        ),
        (
            axes[1],
            "MODE2_PRIMARY_ANOMALY_CLUSTER",
            "Mode 2: IF Dominancy",
        ),
    ]:
        counts = (
            summary.loc[summary[column].fillna("").ne(""), column]
            .astype(str).value_counts()
        )
        if counts.empty:
            axis.text(0.5, 0.5, "No incidents", ha="center", va="center")
            axis.axis("off")
        else:
            axis.pie(
                counts.to_numpy(),
                labels=counts.index.tolist(),
                autopct=lambda value: f"{value:.1f}%",
                startangle=90,
            )
            axis.axis("equal")
        axis.set_title(title)
    figure.suptitle("V9.8.5 Cluster Distribution by Detection Mode")
    figure.tight_layout()
    figure.savefig(
        out_dir / "cesmlc_v985_cluster_distribution_pie.png",
        dpi=160,
        bbox_inches="tight",
    )
    plt.close(figure)
    pie(
        summary["DETECTOR_AGREEMENT"],
        "V9.8.5 IF/Signature Detection Agreement",
        "cesmlc_v985_detector_agreement_pie.png",
    )


def write_mode_comparison_evidence(
    out_dir: Path,
    data: pd.DataFrame,
    args: argparse.Namespace,
) -> None:
    """Write union-aware evidence and a compact mode funnel for compare mode."""
    if args.detection_mode != "compare":
        return
    mode1 = _restore_mode1_view(data)
    mode2 = _build_mode2_view(data, args)
    keep = [
        "ECGI", "FEATURE", "SITE_TYPE", "IS_COW_SITE",
        "MODEL_NORMAL_PERCENTILE", "SOURCE_CHANGE_DAYS",
        "SOURCE_CHANGE_EVENTS", "FEATURE_FINAL_CLUSTER",
        "FEATURE_CLUSTER_SCORE", "DETECTION_METHOD",
        "MODE2_MAPPED_CLUSTER", "KMEANS_CLUSTER",
    ]

    def evidence_rows(
        frame: pd.DataFrame,
        source: str,
    ) -> pd.DataFrame:
        if frame is None or frame.empty:
            return pd.DataFrame()
        selected = frame.loc[
            col_num(frame, "REPORTABLE_EVIDENCE_ROW_FLAG", 0).eq(1),
            [column for column in keep if column in frame.columns],
        ].copy()
        selected["COMPARISON_SOURCE"] = source
        return selected

    union = pd.concat(
        [
            evidence_rows(mode1, MODE1_LABEL),
            evidence_rows(mode2, MODE2_LABEL),
        ],
        ignore_index=True,
        sort=False,
    )
    if not union.empty:
        union = union.sort_values(
            ["ECGI", "COMPARISON_SOURCE", "FEATURE"]
        ).drop_duplicates(["ECGI", "COMPARISON_SOURCE", "FEATURE"])
    csv_write(
        union,
        out_dir / "cesmlc_v985_comparison_union_feature_evidence.csv",
        [*keep, "COMPARISON_SOURCE"],
    )
    mode1_ecgis = set(
        mode1.loc[
            col_num(mode1, "REPORTABLE_EVIDENCE_ROW_FLAG", 0).eq(1), "ECGI"
        ].astype(str)
    ) if not mode1.empty else set()
    mode2_ecgis = set(
        mode2.loc[
            col_num(mode2, "REPORTABLE_EVIDENCE_ROW_FLAG", 0).eq(1), "ECGI"
        ].astype(str)
    ) if not mode2.empty else set()
    funnel = pd.DataFrame([
        {"STAGE": "MODE1_SIGNATURE_DOMINANT", "ECGI_COUNT": len(mode1_ecgis)},
        {"STAGE": "MODE2_IF_DOMINANT", "ECGI_COUNT": len(mode2_ecgis)},
        {"STAGE": "AGREE_IF_AND_SIGNATURE", "ECGI_COUNT": len(mode1_ecgis & mode2_ecgis)},
        {"STAGE": "JUST_SIGNATURE", "ECGI_COUNT": len(mode1_ecgis - mode2_ecgis)},
        {"STAGE": "JUST_IF", "ECGI_COUNT": len(mode2_ecgis - mode1_ecgis)},
        {"STAGE": "COMPARISON_UNION", "ECGI_COUNT": len(mode1_ecgis | mode2_ecgis)},
    ])
    csv_write(funnel, out_dir / "cesmlc_v985_mode_comparison_funnel.csv")


def write_outputs(
    out_dir: Path,
    data: pd.DataFrame,
    candidate_population: pd.DataFrame,
    summary: pd.DataFrame,
    calibration: pd.DataFrame,
    events: pd.DataFrame,
    conflicts: pd.DataFrame,
    run_info: RunInfo,
    site_type_counts: Dict[str, int],
    args: argparse.Namespace,
) -> None:
    header("PHASE 7 - WRITE ROBUST V9.8 OUTPUTS")
    out_dir.mkdir(parents=True, exist_ok=True)
    candidate = data[data["POPULATION"].eq("CANDIDATE")].copy() if not data.empty else pd.DataFrame()
    if not candidate.empty and not summary.empty:
        decision = summary[[
            "ECGI", "INCIDENT_RANK", "PRIMARY_ANOMALY_CLUSTER",
            "ECGI_ANOMALY_RISK_SCORE", "ALL_MATCHED_CLUSTERS",
        ]].rename(columns={
            "PRIMARY_ANOMALY_CLUSTER": "ECGI_FINAL_CLUSTER",
            "ECGI_ANOMALY_RISK_SCORE": "ECGI_FINAL_RISK_SCORE",
        })
        candidate = candidate.merge(decision, on="ECGI", how="left", validate="many_to_one")
    elif not candidate.empty:
        candidate["INCIDENT_RANK"] = np.nan
        candidate["ECGI_FINAL_CLUSTER"] = ""
        candidate["ECGI_FINAL_RISK_SCORE"] = np.nan
        candidate["ALL_MATCHED_CLUSTERS"] = ""
    path_summary = build_change_path_summary(events)
    if not candidate.empty and not path_summary.empty:
        candidate = candidate.merge(
            path_summary[path_summary["POPULATION"].eq("CANDIDATE")],
            on=["POPULATION", "ECGI", "FEATURE"], how="left",
        )
    for column in ["CHANGE_DATES", "SOURCE_TRANSITIONS_BY_DATE"]:
        if not candidate.empty:
            candidate[column] = col_text(candidate, column, "")
    evidence = candidate[
        col_num(candidate, "REPORTABLE_EVIDENCE_ROW_FLAG", 0).eq(1)
    ].copy() if not candidate.empty else pd.DataFrame()
    if not evidence.empty:
        evidence = evidence.sort_values(
            ["INCIDENT_RANK", "FEATURE_CLUSTER_SCORE", "SOURCE_CHANGE_DAYS", "FEATURE"],
            ascending=[True, False, False, True], na_position="last",
        )
    quiet_column = (
        "QUIET_LAST_N_DAYS_FLAG"
        if "QUIET_LAST_N_DAYS_FLAG" in candidate.columns
        else "STABLE_LAST_N_DAYS_FLAG"
    )
    quiet_cells = (
        candidate.loc[col_num(candidate, quiet_column, 0).eq(1)]
        .sort_values("ECGI").groupby("ECGI", as_index=False).head(1)
        if not candidate.empty else pd.DataFrame()
    )
    engineered_ecgis = set(candidate["ECGI"].astype(str)) if not candidate.empty else set()
    unscorable_candidates = (
        candidate_population.loc[
            ~candidate_population["ECGI"].astype(str).isin(engineered_ecgis)
        ].copy()
        if candidate_population is not None and not candidate_population.empty
        else pd.DataFrame()
    )
    if not unscorable_candidates.empty:
        unscorable_candidates["DISPOSITION"] = "NO_NON_NULL_MONITORED_SOURCE_HISTORY"
        unscorable_candidates["DETAIL"] = (
            "Candidate passed the configured mismatch gate but no selected non-PCI source feature "
            "had a usable value in the analysis window."
        )
    issue_summary = (
        summary.groupby("PRIMARY_ANOMALY_CLUSTER", as_index=False).agg(
            IMPACTED_ECGI_COUNT=("ECGI", "nunique"),
            MAX_RISK_SCORE=("ECGI_ANOMALY_RISK_SCORE", "max"),
            MEAN_RISK_SCORE=("ECGI_ANOMALY_RISK_SCORE", "mean"),
            COW_ECGI_COUNT=("IS_COW_SITE", "sum"),
        ) if not summary.empty else pd.DataFrame()
    )
    if not issue_summary.empty:
        issue_summary["CLUSTER_DESCRIPTION"] = issue_summary[
            "PRIMARY_ANOMALY_CLUSTER"
        ].map(cluster_description)
    candidate_conflicts = (
        conflicts.loc[conflicts["POPULATION"].astype(str).eq("CANDIDATE")].copy()
        if conflicts is not None and not conflicts.empty and "POPULATION" in conflicts.columns
        else pd.DataFrame()
    )
    cow_location_diagnostics = (
        candidate.loc[
            col_num(candidate, "IS_COW_SITE", 0).eq(1)
            & candidate["FEATURE"].isin(["ANTENNALAT", "ANTENNALON"])
        ].copy()
        if not candidate.empty and "FEATURE" in candidate.columns
        else pd.DataFrame()
    )
    if not cow_location_diagnostics.empty:
        cow_location_diagnostics = cow_location_diagnostics.sort_values(
            ["SOURCE_CHANGE_DAYS", "SOURCE_CHANGE_EVENTS", "ECGI", "FEATURE"],
            ascending=[False, False, True, True],
        )
    funnel = build_decision_funnel(data, summary, run_info)
    feature_diagnostics = build_feature_detection_diagnostics(data, args)
    representative_reconciliation = build_daily_representative_reconciliation(
        data, run_info, args
    )

    summary_columns = [
        "INCIDENT_RANK", "ECGI", "ECGI_LABEL", "SITE_TYPE", "IS_COW_SITE",
        "PRIMARY_ANOMALY_CLUSTER", "PRIMARY_ABNORMAL_FEATURE",
        "ECGI_ANOMALY_RISK_SCORE", "ALL_MATCHED_CLUSTERS",
        "QUIET_LAST_N_DAYS_FLAG", "WOULD_OLD_STABILITY_OVERRIDE_DROP_FLAG",
    ]
    evidence_columns = [
        "ECGI", "FEATURE", "SITE_TYPE", "IS_COW_SITE", "EVIDENCE_ROLE",
        "FEATURE_FINAL_CLUSTER", "FEATURE_CLUSTER_SCORE", "SOURCE_CHANGE_DAYS",
        "RECENT_CHANGE_DAYS", "CHANGE_DATES", "SOURCE_TRANSITIONS_BY_DATE",
    ]
    candidate_columns = [
        "POPULATION", "ECGI", "FEATURE", "SITE_TYPE", "IS_COW_SITE",
        "QUIET_LAST_N_DAYS_FLAG", "STABLE_LAST_N_DAYS_FLAG", "STABILITY_STATUS",
        "SOURCE_CHANGE_EVENTS", "SOURCE_CHANGE_DAYS", "RECENT_CHANGE_DAYS",
        "BOUNDARY_CHANGE_EVENTS_RECOVERED",
        "RAW_REAL_BACK_AND_FORTH_FLAG", "RAW_HIGH_FREQUENCY_CHANGES_FLAG",
        "RAW_BACKFORTH_AND_HIGH_FREQUENCY_OVERLAP_FLAG",
        "RAW_RARE_RECURRENT_MODEL_FLAG",
        "RAW_RARE_ONE_CHANGE_CONDITIONAL_MODEL_FLAG", "RAW_RARE_CONFLICT_FLAG",
        "RAW_QUALIFIED_FEATURE_FLAG",
        "MODEL_AVAILABLE_FLAG", "MODEL_NORMAL_PERCENTILE",
        "MODEL_ONE_CHANGE_AGE_BUCKET", "MODEL_ONE_CHANGE_REFERENCE_ROWS",
        "MODEL_ONE_CHANGE_CONDITIONAL_AVAILABLE_FLAG",
        "MODEL_ONE_CHANGE_CONDITIONAL_PERCENTILE",
        "MODEL_ONE_CHANGE_CONFORMAL_P_MAX", "MODEL_ONE_CHANGE_TAIL_VOTE_SHARE",
        "IF_DOMINANT_FEATURE_DETECTED_FLAG",
        "IF_DOMINANT_ECGI_DETECTED_FLAG", "IF_DOMINANT_MAX_PERCENTILE",
        "IF_DOMINANT_SELECTED_FEATURE_COUNT", "MODE2_ECGI_RETAINED_FLAG",
        "MODE2_INDEPENDENT_ONE_CHANGE_EXTREME_FEATURE_FLAG",
        "MODE2_INDEPENDENT_ONE_CHANGE_EXTREME_FLAG",
        "FEATURE_FINAL_CLUSTER", "MULTI_FEATURE_CHANGES_FLAG",
        "REPORTABLE_EVIDENCE_ROW_FLAG",
        "WOULD_OLD_STABILITY_OVERRIDE_DROP_FLAG", "DROPPED_BY_RECENCY_FLAG",
    ]
    quiet_columns = [
        "ECGI", "SITE_TYPE", "IS_COW_SITE", "STABILITY_STATUS",
        "QUIET_LAST_N_DAYS_FLAG", "ECGI_RECENT_OBSERVED_DAYS",
        "ECGI_RECENT_CHANGE_DAYS", "ECGI_DAYS_SINCE_LAST_ANY_CHANGE",
        "WOULD_OLD_STABILITY_OVERRIDE_DROP_FLAG",
    ]
    unscorable_columns = [
        "ECGI", "REGION", "MARKETCLUSTER", "MARKET", "SUBMARKET",
        "SERVER_ID", "USID", "SITE_TYPE", "IS_COW_SITE",
        "SITE_TYPE_MATCH_STATUS", "DISPOSITION", "DETAIL",
    ]
    issue_columns = [
        "PRIMARY_ANOMALY_CLUSTER", "IMPACTED_ECGI_COUNT", "MAX_RISK_SCORE",
        "MEAN_RISK_SCORE", "COW_ECGI_COUNT", "CLUSTER_DESCRIPTION",
    ]
    csv_write(
        summary, out_dir / "cesmlc_v98_final_ecgi_incidents_all.csv",
        summary_columns,
    )
    csv_write(
        summary.head(int(args.final_report_top_n)),
        out_dir / "cesmlc_v98_final_ecgi_incidents_top.csv",
        summary_columns,
    )
    csv_write(
        evidence, out_dir / "cesmlc_v98_reportable_feature_evidence.csv",
        evidence_columns,
    )
    csv_write(
        candidate, out_dir / "cesmlc_v98_candidate_decision_audit.csv",
        candidate_columns,
    )
    csv_write(
        quiet_cells, out_dir / "cesmlc_v98_quiet_last_n_days_context_ecgis.csv",
        quiet_columns,
    )
    csv_write(
        unscorable_candidates, out_dir / "cesmlc_v98_unscorable_candidates.csv",
        unscorable_columns,
    )
    csv_write(
        issue_summary, out_dir / "cesmlc_v98_four_cluster_summary.csv",
        issue_columns,
    )
    csv_write(funnel, out_dir / "cesmlc_v98_decision_funnel.csv")
    csv_write(
        feature_diagnostics,
        out_dir / "cesmlc_v98_feature_detection_diagnostics.csv",
    )
    csv_write(
        representative_reconciliation,
        out_dir / "cesmlc_v98_daily_representative_reconciliation.csv",
    )
    csv_write(
        calibration, out_dir / "cesmlc_v98_model_calibration.csv",
        [
            "FEATURE", "STATUS", "NORMAL_ROWS", "FIT_POOL_ROWS",
            "FIT_ROWS_PER_SEED_MIN", "FIT_ROWS_PER_SEED_MAX",
            "CALIBRATION_ROWS", "CANDIDATE_ROWS",
            "CANDIDATE_FEATURE_ROWS_TOTAL", "CANDIDATE_CHANGED_ROWS_SCORED",
            "N_ESTIMATORS",
            "ONE_CHANGE_CALIBRATION_ROWS", "ONE_CHANGE_CALIBRATION_BUCKET_COUNTS",
            "ONE_CHANGE_CALIBRATION_STATUS",
            "MAX_SAMPLES", "MAX_FEATURES", "ENSEMBLE_SEEDS",
            "ENSEMBLE_SEED_COUNT", "CALIBRATION_TAIL_RATE_MEAN",
            "TUNING_REQUESTED", "TUNING_USED", "TUNING_STATUS",
            "TUNING_SYNTHETIC_AUC", "TUNING_CANDIDATES_TESTED", "MODEL_CONTEXT",
        ],
    )
    csv_write(
        candidate_conflicts, out_dir / "cesmlc_v98_same_day_conflicts_evidence_audit.csv",
        [
            "POPULATION", "ECGI", "FEATURE", "LOAD_DATE",
            "REPRESENTATIVE_SOURCE_VALUE", "SOURCE_VALUES_UNIQUE_COUNT",
            "SOURCE_VALUES_WITH_COUNTS", "RAW_ROWS_PER_ECGI_DATE",
        ],
    )
    csv_write(
        cow_location_diagnostics,
        out_dir / "cesmlc_v984_cow_latlon_candidate_diagnostics.csv",
        [
            "ECGI", "SITE_TYPE", "SITE_TYPE_MATCH_STATUS", "IS_COW_SITE",
            "FEATURE", "SOURCE_CHANGE_EVENTS", "SOURCE_CHANGE_DAYS",
            "RECENT_CHANGE_DAYS", "CHANGE_DATES", "SOURCE_TRANSITIONS_BY_DATE",
            "RAW_REAL_BACK_AND_FORTH_FLAG", "RAW_HIGH_FREQUENCY_CHANGES_FLAG",
            "FEATURE_FINAL_CLUSTER", "REPORTABLE_EVIDENCE_ROW_FLAG",
        ],
    )
    if args.detection_mode in {"if_dominant", "compare"}:
        mode2_audit_columns = [
            "ECGI", "IF_DOMINANT_ECGI_DETECTED_FLAG",
            "IF_DOMINANT_FEATURE_DETECTED_FLAG", "MODE2_MAPPED_CLUSTER",
            "MODE2_ECGI_RETAINED_FLAG",
            "MODE2_INDEPENDENT_ONE_CHANGE_EXTREME_FEATURE_FLAG",
            "MODE2_INDEPENDENT_ONE_CHANGE_EXTREME_FLAG",
            "MODEL_NORMAL_PERCENTILE", "IF_DOMINANT_MAX_PERCENTILE",
            "IF_DOMINANT_SELECTED_FEATURE_COUNT", "KMEANS_CLUSTER",
            "KMEANS_DISTANCE_TO_CENTROID", "KMEANS_SELECTED_K",
            "KMEANS_SELECTION_STATUS", "KMEANS_SILHOUETTE",
        ]
        if bool(args.kmeans_rule_compatibility):
            mode2_audit_columns += [
                "KMEANS_MAJORITY_RULE_CLUSTER", "KMEANS_CLUSTER_SIZE",
                "KMEANS_RULE_PURITY", "KMEANS_RULE_COMPATIBILITY",
            ]
        csv_write(
            candidate.loc[
                col_num(
                    candidate, "IF_DOMINANT_ECGI_DETECTED_FLAG", 0
                ).eq(1),
                [column for column in mode2_audit_columns if column in candidate.columns],
            ].copy() if not candidate.empty else pd.DataFrame(),
            out_dir / "cesmlc_v985_if_dominant_kmeans_mapping_audit.csv",
            mode2_audit_columns,
        )
    write_mode_comparison_evidence(out_dir, data, args)
    write_mode_comparison_artifacts(out_dir, summary, args)
    report_path = out_dir / "cesmlc_v98_final_report.md"
    write_markdown_report(report_path, summary, run_info, args)

    processing = run_info.as_dict()
    matched_site_type_count = sum(
        count for status, count in site_type_counts.items()
        if status not in {"UNMATCHED", "AMBIGUOUS"}
    )
    unknown_site_type_count = int(run_info.candidate_count) - int(matched_site_type_count)
    processing.update({
        "input_file": str(Path(args.input).expanduser()) if args.input else "",
        "cell_viewer_data": str(Path(args.cell_viewer_data).expanduser()) if args.cell_viewer_data else "",
        "detection_mode": str(args.detection_mode),
        "if_dominant_percentile": float(args.if_dominant_percentile),
        "kmeans_requested_clusters": int(args.kmeans_clusters),
        "kmeans_auto": bool(args.kmeans_auto),
        "kmeans_auto_max": int(args.kmeans_auto_max),
        "kmeans_rule_compatibility_column": bool(
            args.kmeans_rule_compatibility
        ),
        "kmeans_decision_policy": (
            "DIAGNOSTIC_ONLY_NEVER_ADDS_REMOVES_OR_MAPS_INCIDENTS"
        ),
        "mode2_mapping_priority": (
            "REAL_BACK_AND_FORTH>HIGH_FREQUENCY_CHANGES>"
            "MULTI_FEATURE_CHANGES>RARE_CHANGES_FALLBACK"
        ),
        "mode2_raw_if_admitted_ecgis": (
            int(candidate.loc[
                col_num(
                    candidate, "IF_DOMINANT_ECGI_DETECTED_FLAG", 0
                ).eq(1),
                "ECGI",
            ].nunique()) if not candidate.empty else 0
        ),
        "mode2_refined_retained_ecgis": (
            int(candidate.loc[
                col_num(candidate, "MODE2_ECGI_RETAINED_FLAG", 0).eq(1),
                "ECGI",
            ].nunique()) if not candidate.empty else 0
        ),
        "mode2_independent_extreme_one_change_ecgis": (
            int(candidate.loc[
                col_num(
                    candidate,
                    "MODE2_INDEPENDENT_ONE_CHANGE_EXTREME_FLAG",
                    0,
                ).eq(1),
                "ECGI",
            ].nunique()) if not candidate.empty else 0
        ),
        "candidate_gate": str(args.candidate_gate),
        "candidate_gate_days": int(args.candidate_gate_days or args.days),
        "candidate_mismatch_min_days": int(args.candidate_mismatch_min_days),
        "loaded_evidence_candidate_ecgis": int(
            getattr(args, "loaded_evidence_candidate_ecgis", 0)
        ),
        "candidate_ecgis_kept_out_of_pandas_no_change_or_conflict": int(
            getattr(args, "candidate_ecgis_kept_out_of_pandas", 0)
        ),
        "candidate_pandas_loading_policy": (
            "LOAD_CANDIDATE_FEATURE_ONLY_IF_INTERDAY_CHANGE_OR_SAMEDAY_CONFLICT;"
            "ALL_GATED_CANDIDATES_REMAIN_IN_STREAMED_SITE_TYPE_AUDIT"
        ),
        "site_type_resolution_counts": site_type_counts,
        "site_type_candidate_count": int(run_info.candidate_count),
        "site_type_matched_candidate_count": int(matched_site_type_count),
        "site_type_unknown_or_ambiguous_candidate_count": max(0, unknown_site_type_count),
        "site_type_dropped_candidate_count": 0,
        "monitored_features": sorted(data["FEATURE"].unique().tolist()) if not data.empty else [],
        "physicalcellid_engineered_rows": int(data["FEATURE"].eq("PHYSICALCELLID").sum()) if not data.empty else 0,
        "physicalcellid_dropped_ecgis": 0,
        "engineered_nonpci_candidate_ecgis": (
            int(candidate["ECGI"].nunique()) if not candidate.empty else 0
        ),
        "candidate_ecgis_without_loaded_change_or_conflict_evidence": max(
            0,
            int(run_info.candidate_count)
            - (int(candidate["ECGI"].nunique()) if not candidate.empty else 0),
        ),
        "boundary_anchor_days": int(args.history_anchor_days),
        "boundary_change_events_recovered": int(
            col_num(candidate, "BOUNDARY_CHANGE_EVENTS_RECOVERED", 0).sum()
        ) if not candidate.empty else 0,
        "trailing_recency_context_days": int(args.stable_days),
        "trailing_recency_min_observed_days": int(args.stable_min_observed_days),
        "quiet_last_n_days_candidate_ecgis": (
            int(candidate.loc[col_num(candidate, quiet_column, 0).eq(1), "ECGI"].nunique())
            if not candidate.empty else 0
        ),
        "old_stability_policy_would_drop_ecgis": (
            int(candidate.loc[
                col_num(candidate, "WOULD_OLD_STABILITY_OVERRIDE_DROP_FLAG", 0).eq(1),
                "ECGI",
            ].nunique()) if not candidate.empty else 0
        ),
        "dropped_by_recency_ecgis": (
            int(candidate.loc[
                col_num(candidate, "DROPPED_BY_RECENCY_FLAG", 0).eq(1), "ECGI"
            ].nunique()) if not candidate.empty else 0
        ),
        "high_frequency_min_change_days": int(args.high_frequency_min_change_days),
        "rare_model_percentile": float(args.rare_model_percentile),
        "one_change_model_conformal_alpha": float(args.one_change_model_alpha),
        "one_change_model_min_reference_rows": int(args.min_one_change_calibration_rows),
        "one_change_min_coverage": float(args.one_change_min_coverage),
        "isolation_forest_ensemble_seeds": isolation_seeds(args),
        "isolation_forest_n_estimators_default": int(args.isolation_n_estimators),
        "isolation_forest_max_samples_default": int(args.isolation_max_samples),
        "isolation_forest_max_features_default": float(args.isolation_max_features),
        "isolation_forest_fine_tuning_requested": bool(args.tune_isolation_forest),
        "isolation_forest_tuning_min_synthetic_auc": float(args.isolation_tune_min_auc),
        "isolation_forest_tuning_scope": "INNER_FIT_POOL_ONLY",
        "isolation_forest_outer_calibration_policy": "UNTOUCHED_REAL_NORMAL_HOLDOUT",
        "multi_feature_min_features": int(args.multi_feature_min_features),
        "multi_feature_min_dates": int(args.multi_feature_min_dates),
        "multi_max_dominant_date_share": float(args.multi_max_dominant_date_share),
        "candidate_feature_rows": len(candidate),
        "changed_candidate_feature_rows": int(
            col_num(candidate, "SOURCE_CHANGE_EVENTS", 0).gt(0).sum()
        ) if not candidate.empty else 0,
        "legacy_min_changed_candidate_feature_rows": int(
            col_num(candidate, "LEGACY_MIN_CHANGE_DAYS", 0).gt(0).sum()
        ) if not candidate.empty else 0,
        "legacy_min_only_changed_candidate_feature_rows": int(
            (
                col_num(candidate, "LEGACY_MIN_CHANGE_DAYS", 0).gt(0)
                & col_num(candidate, "SOURCE_CHANGE_DAYS", 0).eq(0)
            ).sum()
        ) if not candidate.empty else 0,
        "modal_only_changed_candidate_feature_rows": int(
            (
                col_num(candidate, "SOURCE_CHANGE_DAYS", 0).gt(0)
                & col_num(candidate, "LEGACY_MIN_CHANGE_DAYS", 0).eq(0)
            ).sum()
        ) if not candidate.empty else 0,
        "modal_legacy_daily_representative_disagreement_days": int(
            col_num(
                candidate, "DAILY_REPRESENTATIVE_DISAGREEMENT_DAYS", 0
            ).sum()
        ) if not candidate.empty else 0,
        "daily_representative_reconciliation_policy": (
            "LEXICAL_MIN_IS_DIAGNOSTIC_ONLY; DAILY_MODAL_REMAINS_SOLE_DECISION_INPUT"
        ),
        "model_scored_changed_candidate_feature_rows": int(
            col_num(candidate, "MODEL_AVAILABLE_FLAG", 0).eq(1).sum()
        ) if not candidate.empty else 0,
        "unchanged_candidate_feature_rows_intentionally_not_model_scored": int((
            col_num(candidate, "SOURCE_CHANGE_EVENTS", 0).eq(0)
            & col_num(candidate, "MODEL_AVAILABLE_FLAG", 0).eq(0)
        ).sum()) if not candidate.empty else 0,
        "reportable_evidence_rows": len(evidence),
        "reportable_ecgi_incidents": len(summary),
        "mode1_signature_dominant_incidents": int(
            col_num(
                summary, "MODE1_SIGNATURE_DOMINANT_DETECTED_FLAG", 0
            ).sum()
        ) if not summary.empty else 0,
        "mode2_if_dominant_incidents": int(
            col_num(summary, "MODE2_IF_DOMINANT_DETECTED_FLAG", 0).sum()
        ) if not summary.empty else 0,
        "detector_agree_incidents": int(
            col_text(summary, "DETECTOR_AGREEMENT", "")
            .eq("AGREE_IF_AND_SIGNATURE").sum()
        ) if not summary.empty else 0,
        "just_signature_incidents": int(
            col_text(summary, "DETECTOR_AGREEMENT", "")
            .eq("JUST_SIGNATURE").sum()
        ) if not summary.empty else 0,
        "just_if_incidents": int(
            col_text(summary, "DETECTOR_AGREEMENT", "")
            .eq("JUST_IF").sum()
        ) if not summary.empty else 0,
        "cow_latlon_evidence_candidate_rows": len(cow_location_diagnostics),
        "cow_latlon_evidence_candidate_ecgis": (
            int(cow_location_diagnostics["ECGI"].nunique())
            if not cow_location_diagnostics.empty else 0
        ),
        "cow_final_incident_ecgis": (
            int(summary.loc[col_num(summary, "IS_COW_SITE", 0).eq(1), "ECGI"].nunique())
            if not summary.empty and "ECGI" in summary.columns else 0
        ),
        "unscorable_candidate_ecgis": len(unscorable_candidates),
        "history_cache_status": getattr(args, "history_cache_status", "UNKNOWN"),
        "final_clusters": FINAL_CLUSTERS,
        "site_type_policy": "ATTACHED_AFTER_DETECTION_OUTPUT_CONTEXT_ONLY",
        "cow_policy": "NO_COW_TRAINING_GROUP_NO_COW_WEIGHT_NO_COW_SUPPRESSION",
        "default_value_policy": "IGNORED",
        "physicalcellid_policy": "REMOVED_BEFORE_HISTORY_ENGINEERING",
        "trailing_quiet_policy": "OUTPUT_CONTEXT_ONLY_NEVER_SUPPRESSES_FULL_WINDOW_EVIDENCE",
        "runtime_seconds": int(time.time() - RUN_START),
    })
    (out_dir / "cesmlc_v98_processing_report.json").write_text(
        json.dumps(processing, indent=2, default=str), encoding="utf-8"
    )
    readme = f"""# CESMLC V9.8.5 three-mode window-mismatch outputs

Selected detection mode: `{args.detection_mode}`.

`signature_dominant` preserves the V9.8.4 decision behavior. `if_dominant`
admits ECGIs at IF percentile {float(args.if_dominant_percentile):.4f}, clusters
the admitted ECGI vectors with diagnostic KMeans, and maps final labels only
with the full-history signature engine. `compare` writes
`cesmlc_v985_mode_comparison.csv`, detector/cluster agreement columns, and
two pie charts unless `--no-comparison-charts` is supplied. KMeans never adds,
removes, or maps an incident.

Open first:
1. `cesmlc_v98_final_report.md`
2. `cesmlc_v98_final_ecgi_incidents_all.csv`
3. `cesmlc_v98_reportable_feature_evidence.csv`
4. `cesmlc_v98_decision_funnel.csv`
5. `cesmlc_v98_feature_detection_diagnostics.csv`
6. `cesmlc_v98_daily_representative_reconciliation.csv`
7. `cesmlc_v98_candidate_decision_audit.csv`
8. `cesmlc_v98_quiet_last_n_days_context_ecgis.csv`
9. `cesmlc_v98_unscorable_candidates.csv`
10. `cesmlc_v98_model_calibration.csv`
11. `cesmlc_v984_cow_latlon_candidate_diagnostics.csv`
12. `cesmlc_v984_candidate_gate_mismatch_day_distribution.csv`
13. `cesmlc_v985_if_dominant_kmeans_mapping_audit.csv` (Mode 2/3)
14. `cesmlc_v985_mode_comparison.csv` (Mode 3)
15. `cesmlc_v985_comparison_union_feature_evidence.csv` (Mode 3)
16. `cesmlc_v985_mode_comparison_funnel.csv` (Mode 3)
17. `cesmlc_v985_cluster_distribution_pie.png` (Mode 3 unless disabled)
18. `cesmlc_v985_detector_agreement_pie.png` (Mode 3 unless disabled)

This is a clean V9.5/V9.6 branch. It has exactly four final clusters and no legacy
default/location/invalid/persistent/rollout cluster logic. In Mode 1, a qualified
signature cannot be removed by a score threshold. In Mode 2, the configured IF
percentile is the admission gate; after admission, structural inspection maps the
label. Only an ordinary one-time RARE fallback is withheld unless the held-out
same-age one-change conformal test independently confirms it as extreme.

Candidate gate: `{args.candidate_gate}` across the ending
`{int(args.candidate_gate_days or args.days)}` date(s), with at least
`{int(args.candidate_mismatch_min_days)}` distinct mismatch day(s). With the default
`window_any_mismatch`, an ECGI is retained when DB_MISMATCH='YES' anywhere in the
analysis window, even if the latest day is no longer mismatched.

For memory safety, all baseline feature histories are loaded, but a candidate-feature
row enters pandas only when it has an inter-day transition or a same-day conflict.
Stable candidate-feature rows cannot qualify for any of the four clusters and remain
in DuckDB. The SITE_TYPE mapping audit is streamed directly from DuckDB and still
contains every gated candidate ECGI.

PHYSICALCELLID is never engineered. SITE_TYPE and IS_COW_SITE are attached only after
all model, cluster, score, and rank decisions. Every unmatched ECGI remains in
processing with SITE_TYPE=UNKNOWN. COW is not a training stratum, penalty, or suppression.
UNKNOWN is retained; it is never an exclusion condition.

The Isolation Forest uses a deterministic multi-seed ensemble. Each seed is calibrated
against the same untouched outer normal holdout. The median percentile supports the
existing Mode-1 rare-pattern rule and is the explicit admission statistic in Mode 2.
With `--tune-isolation-forest`, parameter selection occurs only inside the fit pool;
candidates and the outer holdout are never inspected.
Only candidate feature histories with a real inter-day transition are model-scored;
unchanged rows cannot enter either model decision branch and remain fully available to
the independent current-state conflict and output-audit paths.
The old V9.7 cell-level stability override has been removed. No source change in the
trailing {int(args.stable_days)} days is context only; qualifying evidence anywhere in the
full analysis window remains eligible. `DROPPED_BY_RECENCY` must always be zero.

History engineering loads up to {int(args.history_anchor_days)} pre-window day(s) only for
the predecessor needed by the first in-window transition. Anchor rows never enter the
analysis-window mode, coverage, or observation counts. The cache fingerprints every input
byte and uses a boundary-aware schema, so an older V9.8 cache is invalidated automatically.

`cesmlc_v98_daily_representative_reconciliation.csv` compares the decision-driving daily
mode with V9.6's lexical-MIN daily value while holding the V9.8 population, normalization,
window, and boundary anchor constant. It reports changed-ECGI overlap, MIN-only/modal-only
change days, representative disagreements, and same-day-conflict overlap. Lexical MIN is
diagnostic only and cannot change a model, cluster, score, suppression, or rank. This is not
an exact V9.6 replay because PCI remains excluded and the configured V9.8 coordinate precision
is retained.

One ordinary change remains non-reportable. A one-change model discovery is allowed only
when it is an ultra-tail against held-out normal one-change histories for the same feature
and last-change-age bucket, has at least {int(args.min_one_change_calibration_rows)} reference
rows, passes worst-seed conformal p<={float(args.one_change_model_alpha):.4f}, and meets the
coverage rule. Current-day or repeated same-day
source conflicts are objective RARE_CHANGES evidence rather than audit-only rows.
"""
    (out_dir / "00_READ_ME_V98.md").write_text(readme, encoding="utf-8")
    log(f"Outputs complete. Open: {report_path}", 1)


def _self_test_args() -> argparse.Namespace:
    return argparse.Namespace(
        stable_days=10,
        history_anchor_days=7,
        latlon_decimals=4,
        stable_min_observed_days=8,
        high_frequency_min_change_days=4,
        rare_model_percentile=98.0,
        one_change_model_alpha=0.005,
        min_one_change_calibration_rows=5,
        one_change_min_coverage=0.80,
        model_support_percentile=95.0,
        multi_feature_min_features=3,
        multi_feature_min_dates=2,
        multi_max_dominant_date_share=0.67,
        summary_top_features=8,
        final_report_top_n=100,
        random_state=42,
        isolation_seeds="17,42,73",
        isolation_n_estimators=400,
        isolation_max_samples=2048,
        isolation_max_features=0.80,
        isolation_fit_sample=100000,
        tune_isolation_forest=False,
        isolation_tune_sample=20000,
        isolation_tune_max_configs=8,
        isolation_tune_min_rows=700,
        isolation_tune_min_auc=0.55,
        disable_isolation_forest=False,
        min_feature_model_rows=8,
        calibration_fraction=0.25,
        min_calibration_rows=5,
        model_risk_baseline_percentile=95.0,
        detection_mode="signature_dominant",
        if_dominant_percentile=98.0,
        kmeans_clusters=4,
        kmeans_auto=False,
        kmeans_auto_max=8,
        kmeans_rule_compatibility=False,
        no_comparison_charts=True,
    )


def run_self_tests() -> int:
    header("V9.8.5 THREE-MODE ROBUST FOUR-CLUSTER SELF-TESTS")
    args = _self_test_args()
    checks: List[Tuple[bool, str]] = []

    def check(condition: object, message: str) -> None:
        checks.append((bool(condition), message))

    def row(ecgi: str, feature: str = "ANTENNAOPENING", **updates: Any) -> Dict[str, Any]:
        base: Dict[str, Any] = {
            "POPULATION": "CANDIDATE",
            "ECGI": ecgi,
            "FEATURE": feature,
            "REGION": "TEST",
            "MARKETCLUSTER": "TEST",
            "MARKET": "TEST",
            "SUBMARKET": "TEST",
            "SERVER_ID": "TEST",
            "USID": "TEST",
            "SITE_TYPE": "MACRO",
            "SITE_TYPE_MATCH_STATUS": "DIRECT_UNIQUE",
            "IS_COW_SITE": 0,
            "QUIET_LAST_N_DAYS_FLAG": 0,
            "STABLE_LAST_N_DAYS_FLAG": 0,
            "STABILITY_STATUS": "RECENT_CHANGE_CONTEXT",
            "ECGI_RECENT_OBSERVED_DAYS": 10,
            "ECGI_LATEST_OBSERVED_ON_END_FLAG": 1,
            "ECGI_RECENT_CHANGE_DAYS": 1,
            "ECGI_DAYS_SINCE_LAST_ANY_CHANGE": 0,
            "SOURCE_CHANGE_EVENTS": 0,
            "SOURCE_CHANGE_DAYS": 0,
            "BOUNDARY_CHANGE_EVENTS_RECOVERED": 0,
            "LEGACY_MIN_CHANGE_EVENTS": 0,
            "LEGACY_MIN_CHANGE_DAYS": 0,
            "LEGACY_MIN_ONLY_CHANGE_DAYS": 0,
            "MODAL_ONLY_CHANGE_DAYS": 0,
            "MODAL_AND_LEGACY_CHANGE_DAYS": 0,
            "DAILY_REPRESENTATIVE_DISAGREEMENT_DAYS": 0,
            "RECENT_CHANGE_DAYS": 0,
            "A_B_A_REVERT_COUNT": 0,
            "REVERSE_TRANSITION_COUNT": 0,
            "MODEL_AVAILABLE_FLAG": 1,
            "MODEL_NORMAL_PERCENTILE": 50.0,
            "MODEL_RISK_SCORE": 0.0,
            "MODEL_ONE_CHANGE_CONDITIONAL_PERCENTILE": 0.0,
            "MODEL_ONE_CHANGE_CONFORMAL_P_MAX": 1.0,
            "MODEL_ONE_CHANGE_REFERENCE_ROWS": 0,
            "MODEL_ONE_CHANGE_TAIL_VOTE_SHARE": 0.0,
            "MODEL_ONE_CHANGE_CONDITIONAL_AVAILABLE_FLAG": 0,
            "MODEL_ONE_CHANGE_AGE_BUCKET": "AGE_0_3D",
            "OBSERVED_COVERAGE_RATE": 1.0,
            "LATEST_OBSERVED_ON_END_DATE_FLAG": 1,
            "CURRENT_DAY_SAME_DAY_CONFLICT_FLAG": 0,
            "RECURRENT_SAME_DAY_CONFLICT_FLAG": 0,
            "SAME_DAY_CONFLICT_DAYS": 0,
            "MODE_IMPURITY_PERCENT": 0.0,
            "FIRST_CHANGE_DATE": pd.NaT,
            "LAST_CHANGE_DATE": pd.NaT,
            "LATEST_SOURCE_VALUE": "B",
            "PREVIOUS_DIFFERENT_SOURCE_VALUE": "A",
            "FEATURE_FAMILY": "CONTINUOUS_NUMERIC",
        }
        base.update(updates)
        return base

    def event(ecgi: str, feature: str, date: str) -> Dict[str, Any]:
        return {
            "POPULATION": "CANDIDATE",
            "ECGI": ecgi,
            "FEATURE": feature,
            "LOAD_DATE": pd.Timestamp(date),
        }

    boundary_run = RunInfo(
        window_start=pd.Timestamp("2026-01-01"),
        window_end=pd.Timestamp("2026-01-02"),
        stable_start=pd.Timestamp("2026-01-01"),
        candidate_count=1,
        baseline_count=0,
        candidate_hash="boundary",
        baseline_hash="boundary",
        latest_raw_rows=1,
        latest_distinct_ecgis=1,
        latest_snapshot_ratio=1.0,
        site_type_match_ratio=0.0,
    )
    boundary_con = import_duckdb().connect(":memory:")
    try:
        create_history_tables(boundary_con)
        boundary_con.execute("""
            CREATE TEMPORARY TABLE feature_daily AS
            SELECT * FROM (VALUES
                ('CANDIDATE', 'BOUNDARY', DATE '2025-12-31', 'A', 'A', 1, 1, 'A (1)'),
                ('CANDIDATE', 'BOUNDARY', DATE '2026-01-01', 'B', 'B', 1, 1, 'B (1)'),
                ('CANDIDATE', 'BOUNDARY', DATE '2026-01-02', 'B', 'B', 1, 1, 'B (1)')
            ) t(POPULATION, ECGI, LOAD_DATE, SOURCE_VALUE, LEGACY_MIN_SOURCE_VALUE,
                RAW_ROWS_PER_ECGI_DATE, SOURCE_VALUES_UNIQUE_COUNT,
                SOURCE_VALUES_WITH_COUNTS)
        """)
        insert_feature_history(boundary_con, "CELLTYPE", boundary_run)
        boundary_metric = boundary_con.execute("""
            SELECT OBSERVED_DAYS, FIRST_SOURCE_VALUE, LATEST_SOURCE_VALUE,
                   PREVIOUS_DIFFERENT_SOURCE_VALUE, DISTINCT_SOURCE_VALUES
            FROM feature_history_metrics
        """).fetchone()
        boundary_event = boundary_con.execute("""
            SELECT LOAD_DATE, PREVIOUS_SOURCE_VALUE, CURRENT_SOURCE_VALUE,
                   BOUNDARY_ANCHOR_EVENT_FLAG
            FROM change_events
        """).fetchone()
        check(
            boundary_metric == (2, "B", "B", "A", 1),
            "Pre-window anchor is excluded from all analysis-window history metrics",
        )
        check(
            boundary_event is not None
            and str(boundary_event[0]) == "2026-01-01"
            and boundary_event[1:] == ("A", "B", 1),
            "First analysis-day transition is recovered from the bounded anchor",
        )

        boundary_con.execute("DROP TABLE feature_daily")
        boundary_con.execute("DELETE FROM feature_history_metrics")
        boundary_con.execute("DELETE FROM change_events")
        boundary_con.execute("DELETE FROM same_day_conflicts")
        boundary_con.execute("""
            CREATE TEMPORARY TABLE raw_population_batch AS
            SELECT * FROM (VALUES
                ('CANDIDATE', 'REP_DIFF', DATE '2025-12-31', 'B'),
                ('CANDIDATE', 'REP_DIFF', DATE '2026-01-01', 'A'),
                ('CANDIDATE', 'REP_DIFF', DATE '2026-01-01', 'B'),
                ('CANDIDATE', 'REP_DIFF', DATE '2026-01-01', 'B'),
                ('CANDIDATE', 'REP_DIFF', DATE '2026-01-02', 'B'),
                ('CANDIDATE', 'REP_DIFF', DATE '2026-01-02', 'B')
            ) t(POPULATION, ECGI, LOAD_DATE, CELLTYPE)
        """)
        create_daily_modal_table(boundary_con, "CELLTYPE")
        representative_day = boundary_con.execute("""
            SELECT SOURCE_VALUE, LEGACY_MIN_SOURCE_VALUE,
                   SOURCE_VALUES_UNIQUE_COUNT
            FROM feature_daily
            WHERE LOAD_DATE=DATE '2026-01-01'
        """).fetchone()
        insert_feature_history(boundary_con, "CELLTYPE", boundary_run)
        representative_metric = boundary_con.execute("""
            SELECT LEGACY_MIN_CHANGE_EVENTS, LEGACY_MIN_CHANGE_DAYS,
                   LEGACY_MIN_ONLY_CHANGE_DAYS, MODAL_ONLY_CHANGE_DAYS,
                   MODAL_AND_LEGACY_CHANGE_DAYS,
                   DAILY_REPRESENTATIVE_DISAGREEMENT_DAYS
            FROM feature_history_metrics
        """).fetchone()
        check(
            representative_day == ("B", "A", 2),
            "Daily mode keeps the majority value while the V9.6 audit keeps lexical MIN",
        )
        check(
            representative_metric == (2, 2, 2, 0, 0, 1)
            and boundary_con.execute(
                "SELECT COUNT(*) FROM change_events"
            ).fetchone()[0] == 0,
            "Legacy-MIN-only transitions are reconciled but never enter modal decisions",
        )
    finally:
        boundary_con.close()

    reconciliation_run = RunInfo(
        window_start=pd.Timestamp("2026-01-01"),
        window_end=pd.Timestamp("2026-01-30"),
        stable_start=pd.Timestamp("2026-01-21"),
        candidate_count=4,
        baseline_count=0,
        candidate_hash="reconciliation",
        baseline_hash="reconciliation",
        latest_raw_rows=4,
        latest_distinct_ecgis=4,
        latest_snapshot_ratio=1.0,
        site_type_match_ratio=0.0,
    )
    reconciliation_rows = pd.DataFrame([
        row(
            "REC_LEGACY_ONLY", "CELLTYPE",
            LEGACY_MIN_CHANGE_EVENTS=2,
            LEGACY_MIN_CHANGE_DAYS=2,
            LEGACY_MIN_ONLY_CHANGE_DAYS=2,
            DAILY_REPRESENTATIVE_DISAGREEMENT_DAYS=1,
            SAME_DAY_CONFLICT_DAYS=1,
        ),
        row(
            "REC_BOTH", "CELLTYPE",
            SOURCE_CHANGE_EVENTS=1,
            SOURCE_CHANGE_DAYS=1,
            LEGACY_MIN_CHANGE_EVENTS=1,
            LEGACY_MIN_CHANGE_DAYS=1,
            MODAL_AND_LEGACY_CHANGE_DAYS=1,
        ),
        row(
            "REC_MODAL_ONLY", "CELLTYPE",
            SOURCE_CHANGE_EVENTS=1,
            SOURCE_CHANGE_DAYS=1,
            MODAL_ONLY_CHANGE_DAYS=1,
        ),
    ])
    reconciliation = build_daily_representative_reconciliation(
        reconciliation_rows, reconciliation_run, args
    ).iloc[0]
    check(
        int(reconciliation["CANDIDATE_ECGIS_WITH_NON_NULL_FEATURE_HISTORY"]) == 3
        and int(reconciliation["CANDIDATE_ECGIS_WITHOUT_NON_NULL_FEATURE_HISTORY"]) == 1
        and int(reconciliation["MODAL_CHANGED_ECGIS"]) == 2
        and int(reconciliation["LEGACY_MIN_CHANGED_ECGIS"]) == 2
        and int(reconciliation["BOTH_CHANGED_ECGIS"]) == 1
        and int(reconciliation["LEGACY_MIN_ONLY_CHANGED_ECGIS"]) == 1
        and int(reconciliation["MODAL_ONLY_CHANGED_ECGIS"]) == 1
        and math.isclose(
            float(reconciliation["CHANGED_ECGI_OVERLAP_JACCARD"]), 1.0 / 3.0
        ),
        "Representative reconciliation reports overlap and missing history without changing decisions",
    )

    feature_rows = pd.DataFrame([
        row("HF3", SOURCE_CHANGE_EVENTS=3, SOURCE_CHANGE_DAYS=3, RECENT_CHANGE_DAYS=2),
        row("HF4", SOURCE_CHANGE_EVENTS=4, SOURCE_CHANGE_DAYS=4, RECENT_CHANGE_DAYS=2),
        row("BF", SOURCE_CHANGE_EVENTS=2, SOURCE_CHANGE_DAYS=2,
            A_B_A_REVERT_COUNT=1, RECENT_CHANGE_DAYS=1),
        row("BF_DOUBLE_AUDIT", SOURCE_CHANGE_EVENTS=2, SOURCE_CHANGE_DAYS=2,
            A_B_A_REVERT_COUNT=1, REVERSE_TRANSITION_COUNT=1,
            RECENT_CHANGE_DAYS=1),
        row("RARE_ONE", SOURCE_CHANGE_EVENTS=1, SOURCE_CHANGE_DAYS=1,
            MODEL_NORMAL_PERCENTILE=100.0, RECENT_CHANGE_DAYS=1),
        row("RARE_ONE_CONDITIONAL", SOURCE_CHANGE_EVENTS=1, SOURCE_CHANGE_DAYS=1,
            MODEL_NORMAL_PERCENTILE=100.0, RECENT_CHANGE_DAYS=1,
            MODEL_ONE_CHANGE_CONDITIONAL_PERCENTILE=99.9,
            MODEL_ONE_CHANGE_CONFORMAL_P_MAX=0.001,
            MODEL_ONE_CHANGE_REFERENCE_ROWS=500,
            MODEL_ONE_CHANGE_TAIL_VOTE_SHARE=1.0,
            MODEL_ONE_CHANGE_CONDITIONAL_AVAILABLE_FLAG=1),
        row("RARE_ONE_DISAGREE", SOURCE_CHANGE_EVENTS=1, SOURCE_CHANGE_DAYS=1,
            MODEL_NORMAL_PERCENTILE=100.0, RECENT_CHANGE_DAYS=1,
            MODEL_ONE_CHANGE_CONDITIONAL_PERCENTILE=99.9,
            MODEL_ONE_CHANGE_CONFORMAL_P_MAX=0.020,
            MODEL_ONE_CHANGE_REFERENCE_ROWS=500,
            MODEL_ONE_CHANGE_TAIL_VOTE_SHARE=2.0 / 3.0,
            MODEL_ONE_CHANGE_CONDITIONAL_AVAILABLE_FLAG=1),
        row("RARE_TWO", SOURCE_CHANGE_EVENTS=2, SOURCE_CHANGE_DAYS=2,
            MODEL_NORMAL_PERCENTILE=99.0, RECENT_CHANGE_DAYS=1),
        row("CURRENT_CONFLICT", SOURCE_CHANGE_EVENTS=0, SOURCE_CHANGE_DAYS=0,
            MODEL_NORMAL_PERCENTILE=50.0,
            CURRENT_DAY_SAME_DAY_CONFLICT_FLAG=1, SAME_DAY_CONFLICT_DAYS=1),
        row("ORDINARY_TWO", SOURCE_CHANGE_EVENTS=2, SOURCE_CHANGE_DAYS=2,
            MODEL_NORMAL_PERCENTILE=80.0, RECENT_CHANGE_DAYS=1),
        row("QUIET_HF", SOURCE_CHANGE_EVENTS=5, SOURCE_CHANGE_DAYS=5,
            QUIET_LAST_N_DAYS_FLAG=1, STABLE_LAST_N_DAYS_FLAG=1,
            STABILITY_STATUS="QUIET_LAST_10D_CONTEXT_ONLY",
            RECENT_CHANGE_DAYS=0, ECGI_RECENT_CHANGE_DAYS=0),
        row("QUIET_BF", SOURCE_CHANGE_EVENTS=2, SOURCE_CHANGE_DAYS=2,
            A_B_A_REVERT_COUNT=1, QUIET_LAST_N_DAYS_FLAG=1,
            STABLE_LAST_N_DAYS_FLAG=1,
            STABILITY_STATUS="QUIET_LAST_10D_CONTEXT_ONLY",
            RECENT_CHANGE_DAYS=0, ECGI_RECENT_CHANGE_DAYS=0),
        row("COW_HF", SOURCE_CHANGE_EVENTS=4, SOURCE_CHANGE_DAYS=4,
            RECENT_CHANGE_DAYS=2, SITE_TYPE="COW", IS_COW_SITE=1),
        row("MACRO_HF", SOURCE_CHANGE_EVENTS=4, SOURCE_CHANGE_DAYS=4,
            RECENT_CHANGE_DAYS=2, SITE_TYPE="MACRO", IS_COW_SITE=0),
    ])
    classified = assign_feature_clusters(feature_rows, args)
    indexed = classified.set_index("ECGI")
    check(indexed.loc["HF3", "HIGH_FREQUENCY_CHANGES_FLAG"] == 0,
          "Exactly three change dates do not qualify as high frequency")
    check(indexed.loc["HF4", "HIGH_FREQUENCY_CHANGES_FLAG"] == 1,
          "Four change dates qualify as high frequency")
    check(indexed.loc["BF", "REAL_BACK_AND_FORTH_FLAG"] == 1,
          "A real A->B->A reversal qualifies as back-and-forth")
    check(indexed.loc["BF", "FEATURE_CLUSTER_SCORE"]
          == indexed.loc["BF_DOUBLE_AUDIT", "FEATURE_CLUSTER_SCORE"],
          "One physical reversal is not double-counted by two audit flags")
    check(indexed.loc["RARE_ONE", "RARE_CHANGES_FLAG"] == 0
          and indexed.loc["RARE_ONE", "ONE_DAY_RARE_GUARD_FLAG"] == 1,
          "A pooled percentile of 100 cannot promote an ordinary one-change row")
    check(indexed.loc["RARE_ONE_CONDITIONAL", "RARE_CHANGES_FLAG"] == 1
          and indexed.loc["RARE_ONE_CONDITIONAL", "ONE_DAY_RARE_GUARD_FLAG"] == 0,
          "A worst-seed conformal ultra-tail versus same-age one-change normals can qualify")
    check(indexed.loc["RARE_ONE_DISAGREE", "RARE_CHANGES_FLAG"] == 0,
          "One-change model discovery is rejected when any seed fails the conformal alpha")
    check(indexed.loc["RARE_TWO", "RARE_CHANGES_FLAG"] == 1,
          "Two-date unusual temporal behavior can qualify as rare")
    check(indexed.loc["CURRENT_CONFLICT", "RARE_CHANGES_FLAG"] == 1,
          "A latest-day same-source conflict remains actionable without fake transitions")
    check(indexed.loc["ORDINARY_TWO", "FEATURE_REPORTABLE_FLAG"] == 0,
          "Two ordinary change dates are not automatically abnormal")
    check(indexed.loc["QUIET_HF", "FEATURE_FINAL_CLUSTER"]
          == "HIGH_FREQUENCY_CHANGES"
          and indexed.loc["QUIET_HF", "FEATURE_REPORTABLE_FLAG"] == 1
          and indexed.loc["QUIET_HF", "WOULD_OLD_STABILITY_OVERRIDE_DROP_FEATURE_FLAG"] == 1,
          "Trailing quiet context cannot erase older high frequency")
    check(indexed.loc["QUIET_BF", "FEATURE_FINAL_CLUSTER"]
          == "REAL_BACK_AND_FORTH",
          "Trailing quiet context cannot erase an older reverse transition")
    check(
        indexed.loc["COW_HF", "FEATURE_FINAL_CLUSTER"]
        == indexed.loc["MACRO_HF", "FEATURE_FINAL_CLUSTER"]
        and indexed.loc["COW_HF", "FEATURE_CLUSTER_SCORE"]
        == indexed.loc["MACRO_HF", "FEATURE_CLUSTER_SCORE"],
        "SITE_TYPE/COW context cannot change a cluster or score",
    )

    topology_rows: List[Dict[str, Any]] = []
    topology_events: List[Dict[str, Any]] = []
    patterns = {
        "ASYNC": {
            "CELLTYPE": ["2026-01-22"],
            "GEOTYPE": ["2026-01-25"],
            "SUPPORTEDTECHNOLOGIES": ["2026-01-28"],
        },
        "ASYNC_QUIET": {
            "CELLTYPE": ["2026-01-03"],
            "GEOTYPE": ["2026-01-06"],
            "SUPPORTEDTECHNOLOGIES": ["2026-01-09"],
        },
        "BULK": {
            "CELLTYPE": ["2026-01-28"],
            "GEOTYPE": ["2026-01-28"],
            "SUPPORTEDTECHNOLOGIES": ["2026-01-28"],
        },
        "MAJORITY": {
            "CELLTYPE": ["2026-01-28"],
            "GEOTYPE": ["2026-01-28"],
            "SUPPORTEDTECHNOLOGIES": ["2026-01-25"],
        },
        "TWO_FEATURES": {
            "CELLTYPE": ["2026-01-25"],
            "GEOTYPE": ["2026-01-28"],
        },
        "BULK_PLUS_ASYNC": {
            "CELLTYPE": ["2026-01-20", "2026-01-24"],
            "GEOTYPE": ["2026-01-20", "2026-01-25"],
            "SUPPORTEDTECHNOLOGIES": ["2026-01-20", "2026-01-26"],
            "ANTENNAOPENING": ["2026-01-20", "2026-01-27"],
        },
    }
    for ecgi, feature_dates in patterns.items():
        for feature, dates in feature_dates.items():
            quiet_topology = ecgi == "ASYNC_QUIET"
            topology_rows.append(row(
                ecgi, feature, SOURCE_CHANGE_EVENTS=len(dates),
                SOURCE_CHANGE_DAYS=len(dates),
                RECENT_CHANGE_DAYS=0 if quiet_topology else len(dates),
                QUIET_LAST_N_DAYS_FLAG=int(quiet_topology),
                STABLE_LAST_N_DAYS_FLAG=int(quiet_topology),
                STABILITY_STATUS=(
                    "QUIET_LAST_10D_CONTEXT_ONLY"
                    if quiet_topology else "RECENT_CHANGE_CONTEXT"
                ),
                FEATURE_FAMILY="CATEGORICAL",
            ))
            topology_events.extend(event(ecgi, feature, date) for date in dates)
    topology_input = assign_feature_clusters(pd.DataFrame(topology_rows), args)
    topology = build_multi_feature_topology(
        topology_input, pd.DataFrame(topology_events), args
    )
    topo = topology.groupby("ECGI", as_index=False).head(1).set_index("ECGI")
    check(topo.loc["ASYNC", "MULTI_FEATURE_CHANGES_FLAG"] == 1,
          "Three features on three different dates qualify as multi-feature")
    check(topo.loc["ASYNC_QUIET", "MULTI_FEATURE_CHANGES_FLAG"] == 1,
          "Trailing quiet context cannot erase older asynchronous multi-feature topology")
    check(topo.loc["BULK", "MULTI_FEATURE_CHANGES_FLAG"] == 0,
          "Same-day multi-field bulk change is not multi-feature")
    check(topo.loc["MAJORITY", "MULTI_FEATURE_CHANGES_FLAG"] == 1,
          "A minimal two-date 2+1 topology is not discarded by an over-strict 50% cutoff")
    check(topo.loc["TWO_FEATURES", "MULTI_FEATURE_CHANGES_FLAG"] == 0,
          "Two changed features do not qualify as multi-feature")
    check(topo.loc["BULK_PLUS_ASYNC", "MULTI_FEATURE_CHANGES_FLAG"] == 1,
          "A shared bulk episode cannot hide substantial independent feature-date activity")

    coverage_run = RunInfo(
        window_start=pd.Timestamp("2026-01-01"),
        window_end=pd.Timestamp("2026-01-30"),
        stable_start=pd.Timestamp("2026-01-21"),
        candidate_count=1,
        baseline_count=0,
        candidate_hash="test",
        baseline_hash="test",
        latest_raw_rows=1,
        latest_distinct_ecgis=1,
        latest_snapshot_ratio=1.0,
        site_type_match_ratio=1.0,
    )
    sparse_coverage = pd.DataFrame([
        {
            "POPULATION": "CANDIDATE", "ECGI": "SPARSE", "FEATURE": "CELLTYPE",
            "RECENT_OBSERVED_DAYS": 8, "LATEST_OBSERVED_ON_END_DATE_FLAG": 0,
        },
        {
            "POPULATION": "CANDIDATE", "ECGI": "SPARSE", "FEATURE": "GEOTYPE",
            "RECENT_OBSERVED_DAYS": 1, "LATEST_OBSERVED_ON_END_DATE_FLAG": 1,
        },
    ])
    sparse_context = build_stability_context(
        sparse_coverage,
        pd.DataFrame(columns=["POPULATION", "ECGI", "FEATURE", "LOAD_DATE"]),
        coverage_run,
        args,
    )
    check(sparse_context["STABLE_LAST_N_DAYS_FLAG"].iloc[0] == 0,
          "Sparse features cannot combine separate coverage and latest-day evidence into false stability")

    combined = pd.concat([
        classified,
        topology[topology["ECGI"].eq("ASYNC")],
    ], ignore_index=True, sort=False)
    # Ensure topology-derived reportability is present in the combined fixture.
    combined["REPORTABLE_EVIDENCE_ROW_FLAG"] = (
        col_num(combined, "FEATURE_REPORTABLE_FLAG", 0).eq(1)
        | col_num(combined, "MULTI_FEATURE_CONTRIBUTOR_FLAG", 0).eq(1)
    ).astype(int)
    summary = build_ecgi_summary(combined, args)
    check(summary["ECGI"].is_unique,
          "Final output contains exactly one row per ECGI")
    check(set(summary["PRIMARY_ANOMALY_CLUSTER"]).issubset(set(FINAL_CLUSTERS)),
          "Only the four requested final clusters exist")
    check("QUIET_HF" in set(summary["ECGI"])
          and summary.loc[summary["ECGI"].eq("QUIET_HF"),
                          "PRIMARY_ANOMALY_CLUSTER"].iloc[0]
          == "HIGH_FREQUENCY_CHANGES",
          "Quiet historical abnormalities remain in the final report")
    check(summary.loc[summary["ECGI"].eq("ASYNC"), "PRIMARY_ANOMALY_CLUSTER"].iloc[0]
          == "MULTI_FEATURE_CHANGES",
          "Topology-only asynchronous changes produce a multi-feature incident")

    recall_fixture = finalize_detection(feature_rows, pd.DataFrame(), args)
    check(
        recall_fixture.loc[
            recall_fixture["ECGI"].eq("QUIET_HF"),
            "WOULD_OLD_STABILITY_OVERRIDE_DROP_FLAG",
        ].iloc[0] == 1
        and col_num(recall_fixture, "DROPPED_BY_RECENCY_FLAG", 0).sum() == 0,
        "Old stability-policy losses are audited while dropped-by-recency stays zero",
    )

    detected = detect_features(
        DEFAULT_FEATURES + ["PHYSICALCELLID"],
        ",".join(DEFAULT_FEATURES + ["PHYSICALCELLID"]),
        None,
    )
    check("PHYSICALCELLID" not in detected,
          "PHYSICALCELLID is removed before history engineering")
    check(float(empirical_percentile(np.zeros(20), np.array([0.0]))[0]) == 50.0,
          "A score tied with a constant normal reference calibrates to percentile 50")
    check(not any("SITE" in column or "COW" in column for column in MODEL_INPUT_COLUMNS),
          "SITE_TYPE and COW are structurally absent from the model matrix")

    same_behavior = pd.DataFrame([
        {
            "OBSERVED_DAYS": 30, "SOURCE_CHANGE_EVENTS": 4,
            "SOURCE_CHANGE_DAYS": 4, "DISTINCT_SOURCE_VALUES": 3,
            "MODE_IMPURITY_PERCENT": 20, "ABA_EVENT_RATE": 0.25,
            "REVERSE_EVENT_RATE": 0.25, "RECENT_CHANGE_DAYS": 2,
            "RECENT_OBSERVED_DAYS": 7, "RECENCY_WEIGHTED_ACTIVITY": 2.0,
            "TRANSITION_REPEAT_RATE": 0.25, "CHANGE_INTERVAL_IRREGULARITY": 0.5,
            "SITE_TYPE": "COW", "IS_COW_SITE": 1,
        },
        {
            "OBSERVED_DAYS": 30, "SOURCE_CHANGE_EVENTS": 4,
            "SOURCE_CHANGE_DAYS": 4, "DISTINCT_SOURCE_VALUES": 3,
            "MODE_IMPURITY_PERCENT": 20, "ABA_EVENT_RATE": 0.25,
            "REVERSE_EVENT_RATE": 0.25, "RECENT_CHANGE_DAYS": 2,
            "RECENT_OBSERVED_DAYS": 7, "RECENCY_WEIGHTED_ACTIVITY": 2.0,
            "TRANSITION_REPEAT_RATE": 0.25, "CHANGE_INTERVAL_IRREGULARITY": 0.5,
            "SITE_TYPE": "UNKNOWN", "IS_COW_SITE": 0,
        },
    ])
    model_rows = prepare_model_matrix(same_behavior)
    check(model_rows.iloc[0].equals(model_rows.iloc[1]),
          "Identical temporal behavior has identical model input for COW and UNKNOWN")

    context_input = pd.DataFrame([
        {"ECGI": "A", "FEATURE": "CELLTYPE"},
        {"ECGI": "B", "FEATURE": "CELLTYPE"},
        {"ECGI": "B", "FEATURE": "GEOTYPE"},
    ])
    partial_context = pd.DataFrame([
        {
            "ECGI": "A", "SITE_TYPE": "COW",
            "SITE_TYPE_MATCH_STATUS": "DIRECT_UNIQUE", "IS_COW_SITE": 1,
            "SITE_TYPE_SOURCE_ROW_COUNT": 1, "SITE_TYPE_DISTINCT_COUNT": 1,
        }
    ])
    attached = attach_site_type_context(context_input, partial_context)
    check(
        len(attached) == len(context_input)
        and attached["ECGI"].tolist() == context_input["ECGI"].tolist()
        and attached.loc[attached["ECGI"].eq("B"), "SITE_TYPE"].eq("UNKNOWN").all(),
        "Partial SITE_TYPE coverage retains every row and labels unmatched ECGIs UNKNOWN",
    )
    check(
        stable_feature_seed("CELLTYPE", 42) == stable_feature_seed("CELLTYPE", 42)
        and stable_feature_seed("CELLTYPE", 42) != stable_feature_seed("GEOTYPE", 42),
        "Feature seeds are deterministic and feature-specific",
    )
    order_rows: List[Dict[str, Any]] = []
    for position in range(20):
        order_rows.append({
            "POPULATION": "BASELINE",
            "ECGI": f"BASE_{position:02d}",
            "FEATURE": "CELLTYPE",
            "BASELINE_TRAINING_ELIGIBLE_FLAG": 1,
            "OBSERVED_DAYS": 30,
            "SOURCE_CHANGE_EVENTS": position % 5,
            "SOURCE_CHANGE_DAYS": position % 5,
            "DISTINCT_SOURCE_VALUES": 1 + (position % 3),
            "MODE_IMPURITY_PERCENT": float(position % 7),
            "RECENT_CHANGE_DAYS": position % 2,
            "RECENT_OBSERVED_DAYS": 10,
        })
    order_rows.extend([
        {
            "POPULATION": "CANDIDATE", "ECGI": "ORDER_C1", "FEATURE": "CELLTYPE",
            "BASELINE_TRAINING_ELIGIBLE_FLAG": 0, "OBSERVED_DAYS": 30,
            "SOURCE_CHANGE_EVENTS": 2, "SOURCE_CHANGE_DAYS": 2,
            "DISTINCT_SOURCE_VALUES": 3, "MODE_IMPURITY_PERCENT": 10.0,
            "RECENT_CHANGE_DAYS": 1, "RECENT_OBSERVED_DAYS": 10,
        },
        {
            "POPULATION": "CANDIDATE", "ECGI": "ORDER_C2", "FEATURE": "CELLTYPE",
            "BASELINE_TRAINING_ELIGIBLE_FLAG": 0, "OBSERVED_DAYS": 30,
            "SOURCE_CHANGE_EVENTS": 6, "SOURCE_CHANGE_DAYS": 6,
            "DISTINCT_SOURCE_VALUES": 5, "MODE_IMPURITY_PERCENT": 40.0,
            "RECENT_CHANGE_DAYS": 3, "RECENT_OBSERVED_DAYS": 10,
        },
    ])
    order_args = argparse.Namespace(**vars(args))
    order_args.isolation_n_estimators = 25
    order_args.isolation_max_samples = 8
    order_args.isolation_fit_sample = 20
    ordered_model, _ = add_feature_specific_isolation_forest(
        pd.DataFrame(order_rows), order_args
    )
    shuffled_model, _ = add_feature_specific_isolation_forest(
        pd.DataFrame(order_rows).sample(frac=1.0, random_state=987), order_args
    )
    order_columns = ["ECGI", "MODEL_RAW_ANOMALY_SCORE", "MODEL_NORMAL_PERCENTILE"]
    ordered_candidates = (
        ordered_model.loc[ordered_model["POPULATION"].eq("CANDIDATE"), order_columns]
        .sort_values("ECGI").reset_index(drop=True)
    )
    shuffled_candidates = (
        shuffled_model.loc[shuffled_model["POPULATION"].eq("CANDIDATE"), order_columns]
        .sort_values("ECGI").reset_index(drop=True)
    )
    check(
        ordered_candidates["ECGI"].equals(shuffled_candidates["ECGI"])
        and np.allclose(
            ordered_candidates[["MODEL_RAW_ANOMALY_SCORE", "MODEL_NORMAL_PERCENTILE"]],
            shuffled_candidates[["MODEL_RAW_ANOMALY_SCORE", "MODEL_NORMAL_PERCENTILE"]],
            rtol=0.0, atol=1e-12,
        ),
        "Isolation Forest scores are invariant to upstream row order",
    )
    duplicate_seed_args = argparse.Namespace(isolation_seeds="17,17,42", random_state=42)
    check(isolation_seeds(duplicate_seed_args) == [17, 42],
          "Duplicate ensemble seeds are removed instead of overweighting one model")
    synthetic_a = synthesize_temporal_anomalies(model_rows, 123)
    synthetic_b = synthesize_temporal_anomalies(model_rows, 123)
    check(
        synthetic_a.equals(synthetic_b) and not synthetic_a.equals(model_rows.reset_index(drop=True)),
        "Fine-tuning stress patterns are deterministic and distinct from normal rows",
    )

    # V9.8.5 mode regression and structural-mapping tests.
    legacy_direct = _finalize_signature_dominant(
        feature_rows, pd.DataFrame(), args
    )
    legacy_via_mode = finalize_detection(
        feature_rows, pd.DataFrame(), args
    )
    legacy_columns = list(legacy_direct.columns)
    try:
        pd.testing.assert_frame_equal(
            legacy_direct.reset_index(drop=True),
            legacy_via_mode[legacy_columns].reset_index(drop=True),
            check_dtype=True,
        )
        mode1_equal = True
    except AssertionError:
        mode1_equal = False
    check(
        mode1_equal,
        "Default signature_dominant mode preserves every V9.8.4 decision field",
    )
    check(
        legacy_via_mode["DETECTION_MODE_TAG"].eq(MODE1_LABEL).all(),
        "Mode 1 rows carry the IF + SIGNATURE provenance tag",
    )

    # Mode 2 must work from scored temporal rows with no Mode-1 decisions.
    mode2_multi_rows = pd.DataFrame([
        row("MODE2_MULTI", "CELLTYPE", SOURCE_CHANGE_EVENTS=1, SOURCE_CHANGE_DAYS=1),
        row("MODE2_MULTI", "GEOTYPE", SOURCE_CHANGE_EVENTS=1, SOURCE_CHANGE_DAYS=1),
        row(
            "MODE2_MULTI", "SUPPORTEDTECHNOLOGIES",
            SOURCE_CHANGE_EVENTS=1, SOURCE_CHANGE_DAYS=1,
        ),
    ])
    mode2_multi_events = pd.DataFrame([
        event("MODE2_MULTI", "CELLTYPE", "2026-01-22"),
        event("MODE2_MULTI", "GEOTYPE", "2026-01-25"),
        event("MODE2_MULTI", "SUPPORTEDTECHNOLOGIES", "2026-01-28"),
    ])
    mode_fixture = pd.concat(
        [feature_rows.copy(), mode2_multi_rows], ignore_index=True, sort=False
    )
    mode_fixture["MODEL_AVAILABLE_FLAG"] = 1
    mode_fixture["MODEL_NORMAL_PERCENTILE"] = 50.0
    mode_fixture.loc[
        mode_fixture["ECGI"].isin(
            ["BF", "HF4", "RARE_TWO", "RARE_ONE", "RARE_ONE_CONDITIONAL"]
        ),
        "MODEL_NORMAL_PERCENTILE",
    ] = 99.0
    mode_fixture.loc[
        mode_fixture["ECGI"].eq("MODE2_MULTI")
        & mode_fixture["FEATURE"].eq("CELLTYPE"),
        "MODEL_NORMAL_PERCENTILE",
    ] = 99.0
    mode_fixture.loc[
        mode_fixture["ECGI"].eq("CURRENT_CONFLICT"),
        "MODEL_NORMAL_PERCENTILE",
    ] = 100.0
    mode_args = argparse.Namespace(**vars(args))
    mode_args.detection_mode = "if_dominant"
    mode_args.if_dominant_percentile = 98.0
    mode_args.kmeans_clusters = 4
    mode_args.kmeans_auto = False
    mode_args.kmeans_rule_compatibility = False
    forbidden_mode1_inputs = [
        column for column in mode_fixture.columns
        if column.startswith("MODE1_") or column.startswith("RAW_")
    ]
    mode_fixture = mode_fixture.drop(columns=forbidden_mode1_inputs)
    mode2 = _build_mode2_view(mode_fixture, mode_args, mode2_multi_events)
    mode2_ecgi = mode2.groupby("ECGI", as_index=False).head(1).set_index("ECGI")
    check(
        mode2_ecgi.loc["BF", "MODE2_MAPPED_CLUSTER"] == "REAL_BACK_AND_FORTH"
        and mode2_ecgi.loc["HF4", "MODE2_MAPPED_CLUSTER"]
        == "HIGH_FREQUENCY_CHANGES"
        and mode2_ecgi.loc["MODE2_MULTI", "MODE2_MAPPED_CLUSTER"]
        == "MULTI_FEATURE_CHANGES"
        and mode2_ecgi.loc["RARE_TWO", "MODE2_MAPPED_CLUSTER"]
        == "RARE_CHANGES",
        "Mode 2 IF admission is mapped by BF > HF > MULTI > RARE structural priority",
    )
    check(
        not forbidden_mode1_inputs
        and {
            "MODE2_SIGNATURE_BACK_FORTH_FLAG",
            "MODE2_SIGNATURE_HIGH_FREQUENCY_FLAG",
            "MODE2_SIGNATURE_MULTI_FEATURE_FLAG",
        }.issubset(mode2.columns),
        "Mode 2 performs fresh signature inspection without Mode-1/RAW decision inputs",
    )
    check(
        mode2_ecgi.loc["CURRENT_CONFLICT", "IF_DOMINANT_ECGI_DETECTED_FLAG"] == 0,
        "Conflict-only unscored/no-transition evidence cannot enter IF-dominant mode",
    )
    check(
        mode2_ecgi.loc["RARE_ONE", "IF_DOMINANT_ECGI_DETECTED_FLAG"] == 1
        and mode2_ecgi.loc["RARE_ONE", "MODE2_ECGI_RETAINED_FLAG"] == 0
        and mode2_ecgi.loc[
            "RARE_ONE", "REPORTABLE_EVIDENCE_ROW_FLAG"
        ] == 0,
        "Mode 2 exposes but does not report an ordinary one-time IF admission",
    )
    check(
        mode2_ecgi.loc[
            "RARE_ONE_CONDITIONAL",
            "MODE2_INDEPENDENT_ONE_CHANGE_EXTREME_FLAG",
        ] == 1
        and mode2_ecgi.loc[
            "RARE_ONE_CONDITIONAL", "MODE2_ECGI_RETAINED_FLAG"
        ] == 1
        and mode2_ecgi.loc[
            "RARE_ONE_CONDITIONAL", "MODE2_MAPPED_CLUSTER"
        ] == "RARE_CHANGES",
        "Mode 2 retains an independently calibrated extreme one-time change",
    )
    check(
        "KMEANS_RULE_COMPATIBILITY" not in mode2.columns,
        "KMeans-rule compatibility columns are absent unless requested",
    )
    check(
        math.isclose(
            float(mode2_ecgi.loc["BF", "IF_DOMINANT_MAX_PERCENTILE"]), 99.0
        )
        and scalar_int(
            mode2_ecgi.loc["BF", "IF_DOMINANT_SELECTED_FEATURE_COUNT"]
        ) == 1,
        "Mode 2 exposes its admitting IF maximum and selected-feature count",
    )
    nonconsecutive_fixture = mode_fixture.copy()
    nonconsecutive_fixture.index = np.arange(
        100, 100 + 3 * len(nonconsecutive_fixture), 3
    )
    nonconsecutive_mode2 = _build_mode2_view(
        nonconsecutive_fixture, mode_args, mode2_multi_events
    )
    check(
        len(nonconsecutive_mode2) == len(mode_fixture)
        and set(
            nonconsecutive_mode2.loc[
                nonconsecutive_mode2["IF_DOMINANT_ECGI_DETECTED_FLAG"].eq(1),
                "MODE2_MAPPED_CLUSTER",
            ]
        ) == {
            "REAL_BACK_AND_FORTH",
            "HIGH_FREQUENCY_CHANGES",
            "MULTI_FEATURE_CHANGES",
            "RARE_CHANGES",
        },
        "Mode 2 remains aligned after topology replaces a non-consecutive source index",
    )
    check(
        not build_feature_detection_diagnostics(mode2, mode_args).empty,
        "Mode 2 supplies a complete production feature-diagnostics contract",
    )
    singleton_k = _choose_kmeans_k(
        np.zeros((1, 3)), 4, False, 8, 42
    )
    duplicate_k = _choose_kmeans_k(
        np.zeros((3, 3)), 4, False, 8, 42
    )
    check(
        singleton_k[0] == 1 and duplicate_k[0] == 1,
        "KMeans safely uses K=1 for singleton and identical ECGI vectors",
    )
    compatibility_args = argparse.Namespace(**vars(mode_args))
    compatibility_args.kmeans_rule_compatibility = True
    compatible_mode2 = _build_mode2_view(
        mode_fixture, compatibility_args, mode2_multi_events
    )
    check(
        {
            "KMEANS_RULE_COMPATIBILITY",
            "KMEANS_CLUSTER_SIZE",
            "KMEANS_RULE_PURITY",
        }.issubset(compatible_mode2.columns)
        and compatible_mode2.loc[
            compatible_mode2["IF_DOMINANT_ECGI_DETECTED_FLAG"].eq(1),
            "KMEANS_RULE_PURITY",
        ].between(0, 1).all(),
        "Optional KMeans compatibility includes cluster size and majority-rule purity",
    )
    original_mode1_function = globals()["_finalize_signature_dominant"]
    mode1_called_by_mode2 = False

    def forbidden_mode1_call(*_args: Any, **_kwargs: Any) -> pd.DataFrame:
        nonlocal mode1_called_by_mode2
        mode1_called_by_mode2 = True
        raise AssertionError("Standalone Mode 2 called Mode 1")

    globals()["_finalize_signature_dominant"] = forbidden_mode1_call
    try:
        standalone_mode2 = finalize_detection(
            mode_fixture, mode2_multi_events, mode_args
        )
        standalone_mode2_ok = not standalone_mode2.empty
    finally:
        globals()["_finalize_signature_dominant"] = original_mode1_function
    check(
        standalone_mode2_ok and not mode1_called_by_mode2,
        "Standalone Mode 2 never executes the Mode-1 signature-dominant branch",
    )
    compare_args = argparse.Namespace(**vars(mode_args))
    compare_args.detection_mode = "compare"
    compare_data = finalize_detection(
        mode_fixture, mode2_multi_events, compare_args
    )
    compare_summary = build_ecgi_summary(compare_data, compare_args)
    check(
        not compare_summary.empty
        and compare_summary["ECGI"].is_unique
        and {
            "MODE1_SIGNATURE_DOMINANT_DETECTED_FLAG",
            "MODE2_IF_DOMINANT_DETECTED_FLAG",
            "DETECTOR_AGREEMENT",
            "CLUSTER_AGREEMENT",
        }.issubset(compare_summary.columns),
        "Compare mode returns a one-row ECGI union with detector and cluster agreement",
    )
    check(
        _build_comparison_summary(pd.DataFrame(), compare_args).empty,
        "Compare mode handles an empty candidate dataset",
    )
    check(
        normalize_detection_mode("1") == "signature_dominant"
        and normalize_detection_mode("mode2") == "if_dominant"
        and normalize_detection_mode("both") == "compare",
        "Numeric and friendly mode aliases normalize to canonical modes",
    )

    failed = [message for passed, message in checks if not passed]
    for passed, message in checks:
        log(("PASS: " if passed else "FAIL: ") + message, 1)
    if failed:
        raise AssertionError("V9.8.5 self-tests failed: " + "; ".join(failed))
    log(f"All {len(checks)} robust V9.8.5 tests passed.", 1)
    return 0


def normalize_detection_mode(value: object) -> str:
    normalized = str(value).strip().lower().replace("-", "_")
    aliases = {
        "1": "signature_dominant",
        "mode1": "signature_dominant",
        "mode_1": "signature_dominant",
        "signature": "signature_dominant",
        "signature_dominant": "signature_dominant",
        "2": "if_dominant",
        "mode2": "if_dominant",
        "mode_2": "if_dominant",
        "if": "if_dominant",
        "if_dominant": "if_dominant",
        "3": "compare",
        "mode3": "compare",
        "mode_3": "compare",
        "both": "compare",
        "comparison": "compare",
        "compare": "compare",
    }
    if normalized not in aliases:
        raise argparse.ArgumentTypeError(
            "detection mode must be 1/mode1/signature_dominant, "
            "2/mode2/if_dominant, or 3/mode3/compare"
        )
    return aliases[normalized]


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "CESMLC V9.8.5 three-mode, any-window-mismatch, memory-efficient "
            "four-cluster source-history anomaly detector"
        )
    )
    parser.add_argument("--input", help="CESMLC 30/35-day history CSV")
    parser.add_argument(
        "--cell-viewer-data",
        help=(
            "Optional second-source CSV containing ECGI and SITE_TYPE. Missing/unmatched "
            "context becomes UNKNOWN and never removes an ECGI from detection."
        ),
    )
    parser.add_argument(
        "--strict-site-type-context",
        action="store_true",
        help="Fail on a missing/unreadable SITE_TYPE source instead of continuing with UNKNOWN.",
    )
    parser.add_argument("--out-dir")
    parser.add_argument("--days", type=int, default=30)
    parser.add_argument("--end-date")
    parser.add_argument("--server-id")
    parser.add_argument(
        "--candidate-gate",
        choices=["window_any_mismatch", "latest_mismatch"],
        default="window_any_mismatch",
        help=(
            "window_any_mismatch retains an ECGI when DB_MISMATCH='YES' on any "
            "analysis-window date; latest_mismatch reproduces V9.8.3."
        ),
    )
    parser.add_argument(
        "--candidate-mismatch-min-days",
        type=int,
        default=1,
        help=(
            "Minimum distinct mismatch dates required by window_any_mismatch. "
            "Use 1 for maximum recall."
        ),
    )
    parser.add_argument(
        "--candidate-gate-days",
        type=int,
        default=0,
        help=(
            "Number of ending dates searched by window_any_mismatch; 0 uses --days. "
            "Use 31 with a 31-day input while keeping a 30-day detection window."
        ),
    )
    parser.add_argument("--include-features", default=",".join(DEFAULT_FEATURES))
    parser.add_argument("--exclude-features")
    parser.add_argument(
        "--latlon-decimals", type=int, default=5,
        help=(
            "Coordinate precision before change detection. Five decimals preserves "
            "roughly meter-scale movement; use 4 to suppress small coordinate motion."
        ),
    )
    parser.add_argument("--min-latest-snapshot-ratio", type=float, default=0.80)
    parser.add_argument(
        "--allow-incomplete-latest", action="store_true",
        help="Proceed when the latest snapshot is materially smaller than recent snapshots.",
    )

    parser.add_argument("--stable-days", type=int, default=10)
    parser.add_argument(
        "--history-anchor-days", type=int, default=7,
        help=(
            "Pre-window days loaded only to establish the first in-window transition. "
            "Anchor rows never enter 30-day counts, modes, or observation coverage."
        ),
    )
    parser.add_argument(
        "--stable-min-observed-days", type=int, default=8,
        help=(
            "Minimum observed dates needed to label the trailing window quiet. "
            "This label is context only and never suppresses full-window evidence."
        ),
    )
    parser.add_argument("--high-frequency-min-change-days", type=int, default=2)
    parser.add_argument("--multi-feature-min-features", type=int, default=2)
    parser.add_argument("--multi-feature-min-dates", type=int, default=2)
    parser.add_argument(
        "--multi-max-dominant-date-share", type=float, default=0.67,
        help=(
            "Maximum share of feature-date events occurring on the busiest date for "
            "multi-feature qualification. Same-day-only changes still fail the separate "
            "multi-date and distinct-pattern gates."
        ),
    )
    parser.add_argument(
        "--detection-mode",
        type=normalize_detection_mode,
        choices=DETECTION_MODES,
        default="signature_dominant",
        help=(
            "signature_dominant preserves the V9.8.4 decision behavior; "
            "if_dominant admits ECGIs by an IF percentile then maps them with "
            "the structural signature engine; compare runs both and writes a union."
        ),
    )
    parser.add_argument(
        "--if-dominant-percentile",
        type=float,
        default=98.0,
        help=(
            "Mode-2 ECGI admission threshold on MODEL_NORMAL_PERCENTILE. "
            "At least one genuinely model-scored changed feature must reach it."
        ),
    )
    parser.add_argument(
        "--kmeans-clusters",
        type=int,
        default=4,
        help=(
            "Requested diagnostic KMeans groups for Mode 2/3. It is safely "
            "clamped to the admitted row/unique-vector count."
        ),
    )
    parser.add_argument(
        "--kmeans-auto",
        action="store_true",
        help=(
            "Select diagnostic K using deterministic silhouette evaluation "
            "instead of the fixed --kmeans-clusters value."
        ),
    )
    parser.add_argument(
        "--kmeans-auto-max",
        type=int,
        default=8,
        help="Largest K considered by --kmeans-auto.",
    )
    parser.add_argument(
        "--kmeans-rule-compatibility",
        action="store_true",
        help=(
            "Add KMEANS_RULE_COMPATIBILITY by comparing each ECGI's rule mapping "
            "with the majority rule mapping in its diagnostic KMeans group."
        ),
    )
    parser.add_argument(
        "--no-comparison-charts",
        action="store_true",
        help="In compare mode, write the comparison CSV but skip the two pie-chart PNGs.",
    )

    parser.add_argument(
        "--normal-baseline-max-ecgis", type=int, default=50000,
        help="Normal ECGIs retained for per-feature IF fitting; 50k is memory-safe and ample.",
    )
    parser.add_argument("--min-baseline-coverage", type=float, default=0.70)
    parser.add_argument("--disable-isolation-forest", action="store_true")
    parser.add_argument(
        "--enable-isolation-forest", dest="disable_isolation_forest", action="store_false",
        help="Compatibility flag; IsolationForest is enabled by default.",
    )
    parser.set_defaults(disable_isolation_forest=False)
    parser.add_argument("--isolation-n-estimators", type=int, default=400)
    parser.add_argument("--isolation-max-samples", type=int, default=2048)
    parser.add_argument("--isolation-max-features", type=float, default=0.80)
    parser.add_argument("--isolation-fit-sample", type=int, default=100000)
    parser.add_argument(
        "--isolation-seeds", default="17,42,73",
        help="Comma-separated deterministic ensemble seeds.",
    )
    parser.add_argument(
        "--tune-isolation-forest", action="store_true",
        help=(
            "Fine-tune each feature model inside its outer fit pool using deterministic "
            "synthetic stress patterns; candidates and held-out calibration stay untouched."
        ),
    )
    parser.add_argument("--isolation-tune-sample", type=int, default=20000)
    parser.add_argument("--isolation-tune-max-configs", type=int, default=8)
    parser.add_argument("--isolation-tune-min-rows", type=int, default=700)
    parser.add_argument(
        "--isolation-tune-min-auc", type=float, default=0.55,
        help="Fall back to requested defaults when inner synthetic validation is uninformative.",
    )
    parser.add_argument("--min-feature-model-rows", type=int, default=500)
    parser.add_argument("--calibration-fraction", type=float, default=0.25)
    parser.add_argument("--min-calibration-rows", type=int, default=250)
    parser.add_argument("--model-risk-baseline-percentile", type=float, default=90.0)
    parser.add_argument("--model-support-percentile", type=float, default=95.0)
    parser.add_argument("--rare-model-percentile", type=float, default=98.0)
    parser.add_argument(
        "--one-change-model-alpha", type=float, default=0.005,
        help=(
            "Maximum worst-seed conformal p-value for a one-change model discovery, "
            "calibrated only against held-out normal one-change rows in the same age bucket."
        ),
    )
    parser.add_argument(
        "--min-one-change-calibration-rows", type=int, default=200,
        help="Minimum held-out normal one-change rows in the same feature/age bucket.",
    )
    parser.add_argument(
        "--one-change-min-coverage", type=float, default=0.80,
        help="Minimum analysis-window observation coverage for one-change model discovery.",
    )

    parser.add_argument("--memory-limit", default="8GB")
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--feature-batch-size", type=int, default=2)
    parser.add_argument("--duckdb-file")
    parser.add_argument("--duckdb-temp-dir")
    parser.add_argument("--keep-duckdb", action="store_true")
    parser.add_argument("--reuse-history-cache", action="store_true")
    parser.add_argument("--force-rebuild-history-cache", action="store_true")
    parser.add_argument("--history-cache-dir")
    parser.add_argument(
        "--cache-match-mode", default="compatible",
        choices=["strict", "compatible"],
    )
    parser.add_argument(
        "--cache-input-identity", default="basename",
        choices=["exact", "basename", "none"],
    )
    parser.add_argument("--final-report-top-n", type=int, default=9000)
    parser.add_argument("--summary-top-features", type=int, default=8)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args(argv)


def validate_args(args: argparse.Namespace) -> None:
    if args.self_test:
        return
    if not args.input:
        raise ValueError("--input is required unless --self-test is used.")
    if args.strict_site_type_context and not args.cell_viewer_data:
        raise ValueError("--strict-site-type-context requires --cell-viewer-data.")
    if int(args.days) < int(args.stable_days):
        raise ValueError("--days must be at least --stable-days.")
    if int(args.days) < 2:
        raise ValueError("--days must be at least 2.")
    if not 1 <= int(args.history_anchor_days) <= 31:
        raise ValueError("--history-anchor-days must be between 1 and 31.")
    if not 1 <= int(args.stable_min_observed_days) <= int(args.stable_days):
        raise ValueError("--stable-min-observed-days must be between 1 and --stable-days.")
    if not 0 <= int(args.latlon_decimals) <= 8:
        raise ValueError("--latlon-decimals must be between 0 and 8.")
    if not 0 < float(args.min_latest_snapshot_ratio) <= 1:
        raise ValueError("--min-latest-snapshot-ratio must be in (0, 1].")
    effective_gate_days = int(args.candidate_gate_days or args.days)
    if effective_gate_days < 1:
        raise ValueError("--candidate-gate-days must be 0 or at least 1.")
    if int(args.candidate_mismatch_min_days) < 1:
        raise ValueError("--candidate-mismatch-min-days must be at least 1.")
    if int(args.candidate_mismatch_min_days) > effective_gate_days:
        raise ValueError(
            "--candidate-mismatch-min-days cannot exceed the effective candidate gate window."
        )
    if int(args.high_frequency_min_change_days) < 2:
        raise ValueError("--high-frequency-min-change-days must be at least 2.")
    if int(args.multi_feature_min_features) < 2:
        raise ValueError("--multi-feature-min-features must be at least 2.")
    if int(args.multi_feature_min_dates) < 2:
        raise ValueError("--multi-feature-min-dates cannot be below 2.")
    if not 0 < float(args.multi_max_dominant_date_share) <= 1:
        raise ValueError("--multi-max-dominant-date-share must be in (0, 1].")
    if not 50 <= float(args.if_dominant_percentile) <= 100:
        raise ValueError("--if-dominant-percentile must be in [50, 100].")
    if int(args.kmeans_clusters) < 1:
        raise ValueError("--kmeans-clusters must be at least 1.")
    if int(args.kmeans_auto_max) < 2:
        raise ValueError("--kmeans-auto-max must be at least 2.")
    if (
        args.detection_mode in {"if_dominant", "compare"}
        and bool(args.disable_isolation_forest)
    ):
        raise ValueError(
            "--detection-mode if_dominant/compare requires Isolation Forest; "
            "remove --disable-isolation-forest."
        )
    if not 50 <= float(args.model_support_percentile) <= float(args.rare_model_percentile):
        raise ValueError("--model-support-percentile must not exceed --rare-model-percentile.")
    if not 50 <= float(args.rare_model_percentile) <= 100:
        raise ValueError("--rare-model-percentile must be in [50, 100].")
    if not 0 < float(args.one_change_model_alpha) <= 0.05:
        raise ValueError("--one-change-model-alpha must be in (0, 0.05].")
    minimum_conformal_resolution = (
        int(math.ceil(1.0 / float(args.one_change_model_alpha))) - 1
    )
    if int(args.min_one_change_calibration_rows) < minimum_conformal_resolution:
        raise ValueError(
            "--min-one-change-calibration-rows must be at least "
            f"{minimum_conformal_resolution:,} for "
            f"--one-change-model-alpha={float(args.one_change_model_alpha):g}; "
            "otherwise the finite-sample conformal p-value cannot reach the threshold."
        )
    if not 0 <= float(args.one_change_min_coverage) <= 1:
        raise ValueError("--one-change-min-coverage must be in [0, 1].")
    if not 0 < float(args.calibration_fraction) < 0.5:
        raise ValueError("--calibration-fraction must be between 0 and 0.5.")
    if not 0 <= float(args.min_baseline_coverage) <= 1:
        raise ValueError("--min-baseline-coverage must be in [0, 1].")
    if not 0 < float(args.isolation_max_features) <= 1:
        raise ValueError("--isolation-max-features must be in (0, 1].")
    if not 0.5 <= float(args.isolation_tune_min_auc) <= 1:
        raise ValueError("--isolation-tune-min-auc must be in [0.5, 1].")
    try:
        parsed_seeds = isolation_seeds(args)
    except Exception as exc:
        raise ValueError("--isolation-seeds must be a comma-separated list of integers.") from exc
    if not parsed_seeds:
        raise ValueError("--isolation-seeds must contain at least one integer.")
    if int(args.normal_baseline_max_ecgis) < 0:
        raise ValueError("--normal-baseline-max-ecgis cannot be negative.")
    for name in [
        "isolation_n_estimators", "isolation_max_samples", "isolation_fit_sample",
        "isolation_tune_sample", "isolation_tune_max_configs", "isolation_tune_min_rows",
        "min_feature_model_rows", "min_calibration_rows",
        "min_one_change_calibration_rows", "threads",
        "feature_batch_size", "final_report_top_n", "summary_top_features",
    ]:
        if int(getattr(args, name)) < 1:
            raise ValueError(f"--{name.replace('_', '-')} must be at least 1.")
    if not 0 <= float(args.model_risk_baseline_percentile) < 100:
        raise ValueError("--model-risk-baseline-percentile must be in [0, 100).")


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    validate_args(args)
    if args.self_test:
        return run_self_tests()

    input_path = Path(args.input).expanduser()
    cell_viewer_path = (
        Path(args.cell_viewer_data).expanduser() if args.cell_viewer_data else None
    )
    if not input_path.exists():
        raise FileNotFoundError(f"CESMLC input not found: {input_path}")
    if (
        args.strict_site_type_context
        and cell_viewer_path is not None
        and not cell_viewer_path.exists()
    ):
        raise FileNotFoundError(f"cell_viewer_data not found: {cell_viewer_path}")
    out_dir = Path(args.out_dir).expanduser() if args.out_dir else input_path.parent / "cesmlc_v98_outputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    temp_dir = out_dir / "temp"
    temp_dir.mkdir(parents=True, exist_ok=True)
    duckdb_path = Path(args.duckdb_file).expanduser() if args.duckdb_file else temp_dir / "cesmlc_v98.duckdb"
    duckdb_temp = Path(args.duckdb_temp_dir).expanduser() if args.duckdb_temp_dir else temp_dir / "duckdb_spill"

    header("CESMLC V9.8.5 THREE-MODE FOUR-CLUSTER SOURCE-HISTORY ANOMALY DETECTOR")
    log(f"CESMLC input: {input_path}")
    log(f"SITE_TYPE input: {cell_viewer_path or 'NOT SUPPLIED (UNKNOWN context)'}")
    log(f"Output: {out_dir}")
    log(
        f"Detection mode={args.detection_mode}; "
        f"IF-dominant percentile={float(args.if_dominant_percentile):.4f}; "
        f"KMeans={'auto' if args.kmeans_auto else int(args.kmeans_clusters)} "
        "(diagnostic only)."
    )
    columns, colmap = read_header(input_path)
    date_column = first_existing(columns, DATE_CANDIDATES)
    if not date_column:
        raise ValueError(f"No supported date column found. Tried {DATE_CANDIDATES}.")
    features = detect_features(columns, args.include_features, args.exclude_features)
    if "PHYSICALCELLID" in features:
        raise AssertionError("PHYSICALCELLID survived feature detection.")
    log(f"Monitored source features ({len(features)}): {', '.join(features)}")

    con = open_duckdb(duckdb_path, duckdb_temp, args.threads, args.memory_limit)
    try:
        build_site_type_map(
            con, cell_viewer_path, strict=bool(args.strict_site_type_context)
        )
        run_info = build_populations(con, input_path, colmap, date_column, args)
        site_type_counts = export_candidate_site_context_streaming(
            con,
            run_info,
            out_dir / "cesmlc_v98_site_type_mapping_audit.csv",
        )
        manifest = cache_manifest(input_path, features, run_info, args)
        cache_dir = (
            Path(args.history_cache_dir).expanduser()
            if args.history_cache_dir else out_dir / "history_cache_v98"
        )
        cached = None
        if args.reuse_history_cache and not args.force_rebuild_history_cache:
            cached = load_cache(cache_dir, manifest, args)
        if cached is None:
            if run_info.candidate_count == 0:
                create_history_tables(con)
            else:
                build_history_tables(
                    con, input_path, colmap, date_column, features, run_info, args
                )
            frames = load_history_tables(con)
            if args.reuse_history_cache:
                save_cache(cache_dir, manifest, frames)
                args.history_cache_status = "REBUILT_AND_SAVED"
            else:
                args.history_cache_status = "DISABLED"
        else:
            frames = cached
            args.history_cache_status = "EXACT_CACHE_HIT"
        candidates, population, metrics, changes, conflicts = frames
        args.loaded_evidence_candidate_ecgis = int(
            candidates["ECGI"].nunique()
        ) if not candidates.empty and "ECGI" in candidates.columns else 0
        args.candidate_ecgis_kept_out_of_pandas = max(
            0,
            int(run_info.candidate_count)
            - int(args.loaded_evidence_candidate_ecgis),
        )
        data, events = build_feature_dataset(
            population, metrics, changes, conflicts, run_info, args
        )
        # `frames` otherwise keeps the multi-million-row cached history alive
        # throughout model fitting even though Phase 3 already materialized all
        # required features into `data` and all topology events into `events`.
        del frames, cached, population, metrics, changes
        gc.collect()
        data, calibration = add_feature_specific_isolation_forest(data, args)
        # Normal rows are required through model calibration only. All later
        # decisions and outputs are candidate-only, so release the baseline
        # population before the copy-heavy topology and reporting phases.
        data = data.loc[data["POPULATION"].eq("CANDIDATE")].copy()
        log(
            f"Released baseline feature rows after model calibration; "
            f"candidate rows retained={len(data):,}.",
            1,
        )
        data = finalize_detection(data, events, args) if not data.empty else data
        # SITE_TYPE is attached only after every model score, cluster, and
        # recency-context calculation has been finalized. It is review context only.
        context_ecgis: List[object] = []
        if not data.empty and "ECGI" in data.columns:
            context_ecgis.extend(data["ECGI"].dropna().tolist())
        if not candidates.empty and "ECGI" in candidates.columns:
            context_ecgis.extend(candidates["ECGI"].dropna().tolist())
        candidate_site_context = load_candidate_site_context_subset(
            con, context_ecgis
        )
        data = attach_site_type_context(data, candidate_site_context)
        candidates = attach_site_type_context(candidates, candidate_site_context)
        summary = build_ecgi_summary(data, args)
        write_outputs(
            out_dir, data, candidates, summary, calibration, events, conflicts,
            run_info, site_type_counts, args,
        )
    finally:
        try:
            con.close()
        except Exception:
            pass
        if not args.keep_duckdb:
            shutil.rmtree(temp_dir, ignore_errors=True)
    header("DONE")
    log(f"Open first: {out_dir / 'cesmlc_v98_final_report.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

# pages/1_📥_Ingestion_&_Extraction.py
import streamlit as st
import pandas as pd
from pathlib import Path
from collections import Counter, defaultdict
from datetime import datetime
import random

from streamlit_adapter import (
    iter_files, file_checksum, run_parallel_streaming
)

def use_compact_metrics(value_size="1.35rem", label_size="0.80rem"):
    st.markdown(f"""
    <style>
      div[data-testid="stMetricValue"] {{
        font-size: {value_size};
        line-height: 1.15;
      }}
      div[data-testid="stMetricLabel"] {{
        font-size: {label_size};
        color: rgba(49,51,63,0.7);
      }}
      div[data-testid="stMetricDelta"] svg {{
        width: 12px; height: 12px;
      }}
    </style>
    """, unsafe_allow_html=True)

use_compact_metrics() 

# Left-align all Streamlit buttons (sidebar + main)
st.markdown("""
<style>
/* Applies to both primary and secondary buttons */
.stButton > button,
[data-testid="baseButton-primary"],
[data-testid="baseButton-secondary"] {
  width: 100%;               /* keep full-width if you use use_container_width=True */
  justify-content: flex-start !important;  /* left-align content inside the button */
  text-align: left !important;
  gap: 0.5rem;               /* nicer spacing when you have emojis/icons */
}
</style>
""", unsafe_allow_html=True)


st.set_page_config(page_title="Ingestion & Extraction", layout="wide")

# --------- Sidebar Controls ---------
dataset_mode = st.sidebar.radio(
    "Dataset",
    options=["All", "Sample Demo"],
    index=0,
)

all_root = st.sidebar.text_input(
    "All data root",
    value=str(Path("Data/incoming_full").resolve()),
    help="Top-level folder containing the full corpus"
)

sample_root = st.sidebar.text_input(
    "Sample data root",
    value=str(Path("Data/incoming_sample").resolve()),
    help="Top-level folder containing the curated 5–10 sample files (same structure)"
)

batch_size = st.sidebar.number_input("Batch size", min_value=1, max_value=500, value=5, step=1)
workers = st.sidebar.number_input("Max workers", min_value=1, max_value=64, value=2, step=1)
output_dir = st.sidebar.text_input("Output dir", value=str(Path("Data/Output").resolve()))

col_btn1, col_btn2 = st.sidebar.columns(2)
scan_btn = st.sidebar.button(" 1 Scan Files", use_container_width=True)
process_btn = st.sidebar.button(" 2 Process", use_container_width=True)

def get_active_dirs():
    root = sample_root if dataset_mode == "Sample Demo" else all_root
    return [root]  # keep as list for existing code paths

st.title("Ingestion & Extraction")

# Session state for persistence
if "file_index" not in st.session_state:
    st.session_state.file_index = pd.DataFrame()
if "dupes" not in st.session_state:
    st.session_state.dupes = pd.DataFrame()
if "batch_log" not in st.session_state:
    st.session_state.batch_log = []
if "kpi_accum" not in st.session_state:
    st.session_state.kpi_accum = defaultdict(list)
if "total_files" not in st.session_state:
    st.session_state.total_files = 0
if "batches_total" not in st.session_state:
    st.session_state.batches_total = 0
if "batches_done" not in st.session_state:
    st.session_state.batches_done = 0

# --------- Section 1: Totals, checksums, duplicates ---------
st.subheader("Totals, Checksums & Duplication")

if scan_btn:
    active_dirs = get_active_dirs()
    rows = []
    for fp in iter_files(active_dirs, exts=(".pdf", ".csv")):
        rows.append({
            "path": str(fp),
            "name": fp.name,
            "ext": fp.suffix.lower().replace(".", ""),
            "size_bytes": fp.stat().st_size,
            "checksum_md5": file_checksum(fp),
        })
    idx = pd.DataFrame(rows)
    st.session_state.file_index = idx

    dup_groups = (
        idx.groupby("checksum_md5")
           .filter(lambda g: len(g) > 1)
           .sort_values(["checksum_md5", "name"])
    )
    st.session_state.dupes = dup_groups
## Added
# FX_RATE_FILE = Path(__file__).resolve().parent.parent / "Data" / "fx_rates_sample.csv"

# @st.cache_data(show_spinner=False)
# def load_fx(path: Path) -> pd.DataFrame:
#     return pd.read_csv(path)

# st.subheader("FX Rates (sample)")
# if FX_RATE_FILE.exists():
#     df_fx = load_fx(FX_RATE_FILE)
#     st.dataframe(df_fx, use_container_width=True, height=420)
# else:
#     st.error(f"FX file not found at: {FX_RATE_FILE}")
################
# Display counts
if not st.session_state.file_index.empty:
    idx = st.session_state.file_index
    c1, c2, c3, c4 = st.columns(4)
    total = len(idx)
    pdfs = (idx["ext"] == "pdf").sum()
    csvs = (idx["ext"] == "csv").sum()
    c1.metric("Total Files", f"{total}")
    c2.metric("PDFs", f"{pdfs}")
    c3.metric("CSVs", f"{csvs}")
    c4.metric("Unique Checksums", f"{idx['checksum_md5'].nunique()}")

    with st.expander("📄 File List"):
        st.dataframe(idx.sort_values("name"), use_container_width=True, hide_index=True)

    if st.session_state.dupes.empty:
        st.success("✅ No duplicates found by checksum.")
    else:
        st.warning(f"⚠️ Duplicates found: {st.session_state.dupes['checksum_md5'].nunique()} groups")
        with st.expander("🔁 Duplicate Files (by checksum)"):
            st.dataframe(st.session_state.dupes, use_container_width=True, hide_index=True)
else:
    st.info("Click **Scan Files** to index PDFs/CSVs, compute checksums, and detect duplicates.")

st.divider()

# --------- Section 2: Processing Dashboard — System Health & KPIs ---------
st.subheader("Processing Dashboard (System Health & KPIs)")

kpi_cols = st.columns(5)
kpi_placeholders = {
    "docs_by_type": kpi_cols[0].empty(),
    "in_flight": kpi_cols[1].empty(),
    "avg_ocr": kpi_cols[2].empty(),
    "auto_match": kpi_cols[3].empty(),
    "exceptions_open": kpi_cols[4].empty(),
}
progress_ph = st.empty()
log_table_ph = st.empty()

from collections import Counter
ALLOWED_DOC_TYPES = {"PO", "INV", "GRN"}

def docs_by_type_fixed(counter: Counter) -> str:
    # keep only PO/INV/GRN and show in fixed order
    po  = counter.get("PO", 0)
    inv = counter.get("INV", 0)
    grn = counter.get("GRN", 0)
    return f"PO: {po} | INV: {inv} | GRN: {grn}"

def render_kpis():
    # aggregate from st.session_state.kpi_accum
    acc = st.session_state.kpi_accum
    # Docs by type
    c = Counter()
    for d in acc["doc_counts"]:
        c.update(d)
    doc_str = " | ".join([f"{k}: {v}" for k, v in sorted(c.items())]) if c else "—"
    kpi_placeholders["docs_by_type"].metric("Docs by type", doc_str)

    # In-flight batches
    inflight = st.session_state.batches_total - st.session_state.batches_done
    kpi_placeholders["in_flight"].metric("In-flight batches", f"{max(inflight,0)} running, {st.session_state.batches_done} completed")

    # Avg OCR
    ocr_vals = [v for v in acc["avg_ocr"] if v is not None]
    kpi_placeholders["avg_ocr"].metric("Avg OCR confidence", f"{(sum(ocr_vals)/len(ocr_vals))*100:.1f}%" if ocr_vals else "—")

    # Auto-match rate
    am_vals = [v for v in acc["avg_auto_match"] if v is not None]
    kpi_placeholders["auto_match"].metric("Auto-match rate", f"{(sum(am_vals)/len(am_vals))*100:.1f}%" if am_vals else "—")

    # Exceptions open
    crit = sum(acc["exceptions_critical"]) if acc["exceptions_critical"] else 0
    warn = sum(acc["exceptions_warning"]) if acc["exceptions_warning"] else 0
    kpi_placeholders["exceptions_open"].metric("Exceptions open", f"{crit} critical, {warn} warning")

def append_batch_log(rec):
    st.session_state.batch_log.append({
        "Batch ID": rec["batch_id"],
        "Start (approx)": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"),
        "#Docs": rec["batch_size"],
        "Duration": f"{int(rec['duration_s']//60)}m{int(rec['duration_s']%60)}s",
        "Match rate": f"{(rec['avg_auto_match_rate']*100):.0f}%" if rec["avg_auto_match_rate"] is not None else "—",
        "Exceptions": f"{rec['exceptions_critical']} crit | {rec['exceptions_warning']} warn",
        "Note": f"{' | '.join([f'{k}:{v}' for k,v in (rec['doc_counts'] or {}).items()])}"
    })

def render_batch_table():
    if st.session_state.batch_log:
        df = pd.DataFrame(st.session_state.batch_log)
        log_table_ph.dataframe(df, use_container_width=True, hide_index=True)
    else:
        log_table_ph.info("No batches processed yet.")

# Process button handler
if process_btn:
    st.session_state.batch_log = []
    st.session_state.kpi_accum = defaultdict(list)
    st.session_state.batches_done = 0
    progress_ph.progress(0, text="Starting...")

    active_dirs = get_active_dirs()

    try:
        for evt in run_parallel_streaming(
            input_dirs=active_dirs,        # <— just pass the chosen root
            output_dir=output_dir,
            workers=workers,
            batch_size=batch_size,
        ):
            if evt["event"] == "init":
                st.session_state.total_files = evt.get("total_files", 0)
                st.session_state.batches_total = evt.get("num_batches", 0)
                progress_ph.progress(0, text=f"0/{st.session_state.batches_total} batches")
                render_kpis()

            elif evt["event"] == "batch_done":
                st.session_state.kpi_accum["doc_counts"].append(evt.get("doc_counts", {}))
                st.session_state.kpi_accum["avg_ocr"].append(evt.get("avg_ocr_confidence"))
                st.session_state.kpi_accum["avg_auto_match"].append(evt.get("avg_auto_match_rate"))
                st.session_state.kpi_accum["exceptions_critical"].append(evt.get("exceptions_critical", 0))
                st.session_state.kpi_accum["exceptions_warning"].append(evt.get("exceptions_warning", 0))

                st.session_state.batches_done += 1
                append_batch_log(evt)
                render_kpis()
                render_batch_table()

                pct = int((st.session_state.batches_done / max(st.session_state.batches_total,1)) * 100)
                progress_ph.progress(pct/100.0, text=f"{st.session_state.batches_done}/{st.session_state.batches_total} batches")

        st.success("Processing complete.")
    except Exception as e:
        st.error(f"Pipeline error: {type(e).__name__}: {e}")


# Always render current table
render_batch_table()

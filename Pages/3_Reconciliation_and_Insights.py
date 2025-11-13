# pages/3_Reconciliation_and_Insights.py
from pathlib import Path
import pandas as pd
import streamlit as st

# --- import your reconciler class (adjust module path if needed) ---
from reconcile import ReconcilerV2   # <-- make sure this module exists

st.set_page_config(page_title="Reconciliation & Insights", layout="wide")
st.title("Reconciliation & Insights")

# =========================
# Sidebar: Controls ONLY
# =========================
st.sidebar.header("⚙️ Reconciliation Controls")

tol_total_pct = st.sidebar.number_input(
    "Header tolerance (tol_total_pct)",
    min_value=0.0, max_value=1.0, value=0.01, step=0.001, format="%.3f",
    help="Allowed % deviation between invoice and PO totals (e.g., 0.01 = 1%).",
)
tol_price_pct = st.sidebar.number_input(
    "Unit price tolerance (tol_price_pct)",
    min_value=0.0, max_value=1.0, value=0.02, step=0.001, format="%.3f",
    help="Allowed % deviation between invoice and PO unit price (e.g., 0.02 = 2%).",
)
tol_qty_abs = st.sidebar.number_input(
    "Absolute qty tolerance (tol_qty_abs)",
    min_value=0.0, value=1.0, step=0.5,
    help="Allowed absolute difference in quantities.",
)
desc_match_threshold = st.sidebar.number_input(
    "Description match threshold",
    min_value=0.0, max_value=1.0, value=0.80, step=0.05, format="%.2f",
    help="Similarity threshold for matching descriptions.",
)
prefer_usd = st.sidebar.checkbox(
    "Prefer USD as base currency",
    value=True
)

# =========================
# Main area: Section 1 — Execute reconciliation
# =========================
st.subheader("Execute Reconciliation")

# Where to read Page-1 outputs and where to save Page-3 results
output_dir = st.text_input(
    "Processed output directory",
    value=str(Path("Data/Output").resolve()),
    help="Folder that contains Page-1 outputs (po_agg_df.csv, po_line_df.csv, inv_agg_df.csv, inv_line_df.csv, grn_line_df.csv). "
         "Reconciliation outputs will also be saved here."
)

def _load_csv(path: Path, name: str) -> pd.DataFrame | None:
    if not path.exists():
        st.error(f"Missing file: {path.name} (expected in {path.parent})")
        return None
    try:
        return pd.read_csv(path)
    except Exception as e:
        st.error(f"Failed to read {name}: {type(e).__name__}: {e}")
        return None

# show availability status
outp = Path(output_dir)
po_agg_path  = outp / "po_agg_df.csv"
po_line_path = outp / "po_line_df.csv"
inv_agg_path = outp / "inv_agg_df.csv"
inv_line_path= outp / "inv_line_df.csv"
grn_line_path= outp / "grn_line_df.csv"

with st.expander("Expected inputs (from Page 1)"):
    cols = st.columns(5)
    for c, p in zip(cols, [po_agg_path, po_line_path, inv_agg_path, inv_line_path, grn_line_path]):
        c.write(f"`{p.name}`")
        c.caption("✅ Found" if p.exists() else "❌ Not found")

# Reconcile button in MAIN area (not sidebar)
reconcile_clicked = st.button("🔄 Reconcile", use_container_width=True)

# Keep results in session so they persist while you tweak
if "recon_results_df" not in st.session_state:
    st.session_state.recon_results_df = None
if "exceptions_df" not in st.session_state:
    st.session_state.exceptions_df = None

if reconcile_clicked:
    # Load inputs
    po_agg_df   = _load_csv(po_agg_path,  "po_agg_df")
    po_lines_df = _load_csv(po_line_path, "po_line_df")
    inv_agg_df  = _load_csv(inv_agg_path, "inv_agg_df")
    inv_lines_df= _load_csv(inv_line_path,"inv_line_df")
    grn_lines_df= _load_csv(grn_line_path,"grn_line_df")

    po_agg_df = po_agg_df[po_agg_df['total_amount_usd'].notna()].reset_index(drop=True)
    po_lines_df = po_lines_df[po_lines_df['line_total_usd'].notna()].reset_index(drop=True)
    inv_agg_df = inv_agg_df[inv_agg_df['total_amount_usd'].notna()].reset_index(drop=True)
    inv_lines_df = inv_lines_df[inv_lines_df['line_total_usd'].notna()].reset_index(drop=True)

    # Stop if any missing
    # Stop if any missing (use identity checks, not equality)
    loaded = {
        "po_agg_df": po_agg_df,
        "po_line_df": po_lines_df,
        "inv_agg_df": inv_agg_df,
        "inv_line_df": inv_lines_df,
        "grn_line_df": grn_lines_df,
    }
    missing = [name for name, df in loaded.items() if df is None]
    if missing:
        st.error(f"Missing required inputs: {', '.join(missing)}")
        st.stop()

    # Optional: warn if any inputs are empty
    empties = [name for name, df in loaded.items() if df is not None and df.empty]
    if empties:
        st.warning(f"Empty inputs detected (will proceed): {', '.join(empties)}")


    grn_lines_df['po_number'] = grn_lines_df['po_number'].astype(str).str.replace('O', '0', regex=False)
    grn_lines_df = grn_lines_df.drop_duplicates(subset=['grn_number','line_number','description','quantity']).reset_index(drop=True)

    # Run reconciliation
    try:
        recon = ReconcilerV2(
            tol_total_pct=tol_total_pct,
            tol_price_pct=tol_price_pct,
            tol_qty_abs=tol_qty_abs,
            desc_match_threshold=desc_match_threshold,
            prefer_usd=prefer_usd
        )

        recon_results_df, exceptions_df = recon.reconcile_all(
            po_agg_df=po_agg_df,
            po_lines_df=po_lines_df,
            grn_lines_df=grn_lines_df,
            inv_agg_df=inv_agg_df,
            inv_lines_df=inv_lines_df
        )

        # Save outputs back to output_dir
        outp.mkdir(parents=True, exist_ok=True)
        rr_path = outp / "recon_results_df.csv"
        ex_path = outp / "exceptions_df.csv"
        recon_results_df.to_csv(rr_path, index=False)
        exceptions_df.drop_duplicates().to_csv(ex_path, index=False)

        st.session_state.recon_results_df = recon_results_df
        st.session_state.exceptions_df = exceptions_df.drop_duplicates().reset_index(drop=True)

        st.success(f"Reconciliation complete. Saved: `{rr_path.name}`, `{ex_path.name}`")
    except Exception as e:
        st.error(f"Reconciliation failed: {type(e).__name__}: {e}")

st.divider()

# =========================
# Main area: Section 2 — Show results (scrollable)
# =========================
st.subheader("Results")

tabs = st.tabs(["Reconciliation Results", "Exceptions"])
with tabs[0]:
    st.caption("Joined & matched view (PO ↔ GRN ↔ INV), split by overall status.")

    df = st.session_state.recon_results_df
    if df is None or df.empty:
        st.info("Run reconciliation to see results here.")
    else:
        # case-insensitive check for 'MATCHED'
        status_series = df.get("overall_status")
        if status_series is None:
            st.warning("Column `overall_status` not found in reconciliation results.")
        else:
            is_matched = status_series.astype(str).str.upper().eq("MATCHED")
            df_matched = df[is_matched].reset_index(drop=True)
            df_matched = df_matched.drop_duplicates().reset_index(drop=True)
            df_needs_attention = df[~is_matched].reset_index(drop=True)
            df_needs_attention = df_needs_attention.drop_duplicates().reset_index(drop=True)

            st.markdown(f"**Matched**  ·  {len(df_matched)} rows")
            st.dataframe(df_matched, use_container_width=True, height=300)

            st.markdown("---")

            st.markdown(f"**Needs Attention**  ·  {len(df_needs_attention)} rows")
            st.dataframe(df_needs_attention, use_container_width=True, height=300)


with tabs[1]:
    st.caption("Open issues / rule-triggered exceptions from reconciliation.")
    if st.session_state.exceptions_df is not None and not st.session_state.exceptions_df.empty:
        st.dataframe(st.session_state.exceptions_df, use_container_width=True, height=420)
    else:
        st.info("Run reconciliation to see exceptions here.")

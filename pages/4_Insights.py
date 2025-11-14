# pages/4_Insights.py
from pathlib import Path
import pandas as pd
import streamlit as st
from exception_insight import add_llm_exception_descriptions

import html  # near your imports

def _render_bullets(df_view, font_size="1.05rem"):
    """Render each row as a bullet: EXC_TYPE : Description with stable spacing & font."""
    if df_view is None or df_view.empty:
        st.info("No exception insights for this selection.")
        return

    # CSS: bigger font, normal (non-italic), comfy line-height
    st.markdown(f"""
    <style>
      ul.insight-list {{
        margin: 0.25rem 0 0.5rem 1.1rem;
        padding: 0;
      }}
      ul.insight-list > li {{
        font-size: {font_size};
        line-height: 1.6;
        font-style: normal;
        font-weight: 400;
        list-style-type: disc;
        margin: 0.15rem 0;
      }}
      ul.insight-list > li > strong {{
        font-weight: 600;
        font-style: normal;
      }}
    </style>
    """, unsafe_allow_html=True)

    items = []
    for _, r in df_view.iterrows():
        exc  = html.escape(str(r.get("exception_type", "")).strip())
        desc = html.escape(str(r.get("Final_Exception_Description", "")).strip())
        items.append(f"<li><strong>{exc}</strong>: {desc}</li>")

    st.markdown(f"<ul class='insight-list'>{''.join(items)}</ul>", unsafe_allow_html=True)

image_path = Path(__file__).resolve().parent.parent / "ema_logo.png"

# --- 2. Create Columns for Logo and Title ---
# col1, col2 = st.columns([1, 6]) # Adjust ratio [1, 6] as needed
col1, col2 = st.columns(2)


# --- 3. Add Logo to First Column ---
if image_path.exists():
    with col1:
        st.image(str(image_path), width=100) # Adjust width as needed
else:
    with col1:
        pass # Or just pass

st.set_page_config(page_title="Insights", layout="wide")
st.title("Insights")

# --------------------------
# IMPORT: description generator
# --------------------------
# Change this import to match your script. It must expose a function:
#   build_descriptions(exceptions_df: pd.DataFrame) -> pd.DataFrame
# that returns a DataFrame including a string column "Final_Exception_description"
# try:
#     from insights_generator import build_descriptions  # <-- CHANGE IF NEEDED
# except Exception:
#     build_descriptions = None  # fallback will be used


# --------------------------
# Helpers
# --------------------------
def load_csv(path: Path) -> pd.DataFrame | None:
    try:
        return pd.read_csv(path) if path.exists() else None
    except Exception as e:
        st.error(f"Failed reading {path.name}: {type(e).__name__}: {e}")
        return None

def normalize_status(s: pd.Series) -> pd.Series:
    return s.astype(str).str.strip().str.upper()

@st.cache_data(show_spinner=False, ttl=3600)
def _generate_insights_for_doc(ex_df: pd.DataFrame, id_col: str, doc_id: str) -> list[str]:
    """
    Filter exceptions to one doc, call LLM describer on the small slice,
    and return a list of descriptions. Cached per (id_col, doc_id, hash of slice).
    """
    if ex_df is None or ex_df.empty or id_col not in ex_df.columns or not doc_id:
        return []

    sub = ex_df.loc[ex_df[id_col].astype(str) == str(doc_id)].copy()
    if sub.empty:
        return []

    # Run user’s generator ONLY on this subset
    if add_llm_exception_descriptions is not None:
        try:
            out = add_llm_exception_descriptions(sub)
            if isinstance(out, pd.DataFrame) and "Final_Exception_description" in out.columns:
                vals = out["Final_Exception_description"].astype(str).str.strip()
            else:
                # generator returned unexpected shape; fall back
                vals = _fallback_descriptions(sub)
        except Exception as e:
            st.warning(f"Insight generator failed for {doc_id}; using fallback. {type(e).__name__}: {e}")
            vals = _fallback_descriptions(sub)
    else:
        vals = _fallback_descriptions(sub)

    # Deduplicate while preserving order
    seen, res = set(), []
    for t in vals:
        if t and t not in seen:
            seen.add(t); res.append(t)
    return res

def _fallback_descriptions(df: pd.DataFrame) -> pd.Series:
    # common columns first
    for c in ["exception_message", "exception_type", "message", "reason", "rule_name", "rule_msg"]:
        if c in df.columns:
            return df[c].astype(str).str.strip()
    # very last resort
    cols = [c for c in df.columns if c.lower() in {"invoice_number", "po_number", "exception_severity", "exception_code"}]
    return df[cols].astype(str).agg(" | ".join, axis=1) if cols else pd.Series(["No details"], index=df.index)

@st.cache_data(show_spinner=False, ttl=3600)
def _describe_df_for_doc(ex_df: pd.DataFrame, id_col: str, doc_id: str) -> pd.DataFrame:
    """
    Return a compact, de-duplicated view with just:
      [exception_type, Final_Exception_Description]
    for the selected invoice/PO.
    """
    EMPTY = pd.DataFrame(columns=["exception_type", "Final_Exception_Description"])

    if ex_df is None or ex_df.empty or id_col not in ex_df.columns or not doc_id:
        return EMPTY

    sub = ex_df.loc[ex_df[id_col].astype(str) == str(doc_id)].copy()
    if sub.empty:
        return EMPTY

    # Run your generator ONLY on this subset
    try:
        if add_llm_exception_descriptions is not None:
            out = add_llm_exception_descriptions(sub)
        else:
            out = sub
    except Exception:
        out = sub

    # Normalize description column name (handle casing variants)
    desc_candidates = [
        "Final_Exception_Description",
        "Final_Exception_description",
        "final_exception_description",
        "Final_ExceptionDescription",
    ]
    desc_col = next((c for c in desc_candidates if c in out.columns), None)

    # Build a description if the generator didn’t create one
    if desc_col is None:
        if "message" in out.columns:
            out["Final_Exception_Description"] = out["message"].astype(str)
        elif "exception_type" in out.columns:
            out["Final_Exception_Description"] = out["exception_type"].astype(str)
        else:
            out["Final_Exception_Description"] = out.astype(str).agg(" | ".join, axis=1)
        desc_col = "Final_Exception_Description"

    # Ensure exception_type exists
    if "exception_type" not in out.columns:
        out["exception_type"] = ""

    # Keep only the two columns, drop duplicates, tidy
    view = (
        out[["exception_type", desc_col]]
        .rename(columns={desc_col: "Final_Exception_Description"})
        .astype({"exception_type": "string", "Final_Exception_Description": "string"})
        .drop_duplicates()
        .sort_values(["exception_type", "Final_Exception_Description"], na_position="last")
        .reset_index(drop=True)
    )
    return view




def status_counts(df: pd.DataFrame, id_col: str) -> dict:
    if df is None or df.empty or "overall_status" not in df.columns:
        return {"MATCHED": 0, "PARTIAL": 0, "MISMATCH": 0}
    s = normalize_status(df["overall_status"])
    # Count unique documents by status (not rows)
    grp = df.assign(_status=s).dropna(subset=[id_col])
    counts = grp.groupby([id_col, "_status"]).size().reset_index().pivot_table(
        index=id_col, columns="_status", values=0, fill_value=0
    )
    out = {k: int((counts.get(k) > 0).sum()) for k in ["MATCHED", "PARTIAL", "MISMATCH"]}
    return out

def list_ids_by_status(df: pd.DataFrame, id_col: str, wanted_status: str) -> list[str]:
    if df is None or df.empty or "overall_status" not in df.columns or id_col not in df.columns:
        return []
    wanted = wanted_status.upper()
    mask = normalize_status(df["overall_status"]).eq(wanted)
    ids = (
        df.loc[mask, id_col]
        .dropna()
        .astype(str)
        .drop_duplicates()
        .sort_values()
        .tolist()
    )
    return ids

def describe_for_id(ex_df: pd.DataFrame, id_col: str, doc_id: str) -> list[str]:
    if ex_df is None or ex_df.empty or id_col not in ex_df.columns:
        return []
    tmp = ex_df.loc[ex_df[id_col].astype(str) == str(doc_id)]
    if tmp.empty:
        return []
    # Prefer the generated description column
    desc_col = "Final_Exception_Description" if "Final_Exception_Description" in tmp.columns else None
    if desc_col:
        vals = tmp[desc_col].astype(str).str.strip()
    else:
        # fallback: try 'exception_message' or similar columns
        for c in ["exception_message", "message", "reason"]:
            if c in tmp.columns:
                vals = tmp[c].astype(str).str.strip()
                break
        else:
            vals = tmp.astype(str).agg(" | ".join, axis=1)
    # Deduplicate while preserving order
    seen, out = set(), []
    for t in vals:
        if t and t not in seen:
            seen.add(t); out.append(t)
    return out


# --------------------------
# Inputs (no sidebar on this page)
# --------------------------
output_dir = st.text_input(
    "Processed output directory",
    value=str(Path("Data/Output").resolve()),
    help="Folder that contains recon_results_df.csv and exceptions_df.csv (from Pages 3/1).",
)

outp = Path(output_dir)
recon_path = outp / "recon_results_df.csv"
ex_path    = outp / "exceptions_df.csv"

# Prefer in-memory results if available from Page 3
recon_df = st.session_state.get("recon_results_df")
ex_df    = st.session_state.get("exceptions_df")

if recon_df is None:
    recon_df = load_csv(recon_path)
if ex_df is None:
    ex_df = load_csv(ex_path)

# --------------------------
# Two tabs side-by-side: Invoices, POs
# --------------------------
tabs = st.tabs(["Invoices", "POs"])

# ======= Invoices tab =======
with tabs[0]:
    st.subheader("Invoices", divider=False)

    inv_id_col = "invoice_number"
    if recon_df is None or inv_id_col not in recon_df.columns:
        st.warning(f"`{inv_id_col}` not found in reconciliation results.")
    else:
        counts = status_counts(recon_df, inv_id_col)
        c1, c2, c3 = st.columns(3)
        c1.metric("MATCHED", counts["MATCHED"])
        c2.metric("PARTIAL MATCH", counts["PARTIAL"])
        c3.metric("MISMATCHED", counts["MISMATCH"])

        st.write("")  # spacing
        selected_status = st.selectbox("Status", options=["MISMATCH", "PARTIAL", "MATCHED"], index=0, key="inv_status_sel")
        inv_options = list_ids_by_status(recon_df, inv_id_col, selected_status)
        inv_choice = st.selectbox("Invoice", options=inv_options or ["— none —"], index=0, key="inv_choice", disabled=not bool(inv_options))

        st.write("")  # spacing
        st.markdown("**Insight**")
        btn_disabled = not (inv_options and inv_choice and inv_choice != "— none —")
        gen_inv = st.button("🪄  Generate insights", use_container_width=True, disabled=btn_disabled, key="gen_inv")

        # Persist last generated table per invoice
        st.session_state.setdefault("inv_insight_doc", None)
        st.session_state.setdefault("inv_insight_df", None)

        if gen_inv:
            with st.spinner("Generating insight…"):
                df_view = _describe_df_for_doc(ex_df, inv_id_col, inv_choice)
            st.session_state.inv_insight_doc = inv_choice
            st.session_state.inv_insight_df = df_view.drop_duplicates(subset=['exception_type']).reset_index(drop=True)

        # Show the last generated table if available and matches current selection
        if st.session_state.inv_insight_df is not None and st.session_state.inv_insight_doc == inv_choice:
            if st.session_state.inv_insight_df is not None and not st.session_state.inv_insight_df.empty:
                _render_bullets(st.session_state.inv_insight_df)
            else:
                st.info("No exception insight for this invoice (it might be fully matched).")

        else:
            if btn_disabled:
                st.caption("Select a status and an invoice, then click **Generate insights**.")
            else:
                st.caption("Click **Generate insights** to run the describer for this invoice.")




# ======= POs tab =======
with tabs[1]:
    st.subheader("POs", divider=False)

    po_id_col = "po_number"
    if recon_df is None or po_id_col not in recon_df.columns:
        st.warning(f"`{po_id_col}` not found in reconciliation results.")
    else:
        counts = status_counts(recon_df, po_id_col)
        c1, c2, c3 = st.columns(3)
        c1.metric("MATCHED", counts["MATCHED"])
        c2.metric("PARTIAL MATCH", counts["PARTIAL"])
        c3.metric("MISMATCHED", counts["MISMATCH"])

        st.write("")
        selected_status_po = st.selectbox("Status", options=["MISMATCH", "PARTIAL", "MATCHED"], index=0, key="po_status_sel")
        po_options = list_ids_by_status(recon_df, po_id_col, selected_status_po)
        po_choice = st.selectbox("PO", options=po_options or ["— none —"], index=0, key="po_choice", disabled=not bool(po_options))

        st.write("")
        st.markdown("**Insight**")
        btn_disabled_po = not (po_options and po_choice and po_choice != "— none —")
        gen_po = st.button("🪄  Generate insights", use_container_width=True, disabled=btn_disabled_po, key="gen_po")

        # Persist last generated table per PO
        st.session_state.setdefault("po_insight_doc", None)
        st.session_state.setdefault("po_insight_df", None)

        if gen_po:
            with st.spinner("Generating insight…"):
                df_view = _describe_df_for_doc(ex_df, po_id_col, po_choice)
            st.session_state.po_insight_doc = po_choice
            st.session_state.po_insight_df = df_view.drop_duplicates(subset=['exception_type']).reset_index(drop=True)

        # Show the last generated table if available and matches current selection
        if st.session_state.po_insight_df is not None and st.session_state.po_insight_doc == po_choice:
            if st.session_state.po_insight_df is not None and not st.session_state.po_insight_df.empty:
                _render_bullets(st.session_state.po_insight_df)
            else:
                st.info("No exception insight for this PO (it might be fully matched).")

        else:
            if btn_disabled_po:
                st.caption("Select a status and a PO, then click **Generate insights**.")
            else:
                st.caption("Click **Generate insights** to run the describer for this PO.")






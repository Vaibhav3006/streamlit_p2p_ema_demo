# pages/2_Document_Viewer.py
import base64
from pathlib import Path
import streamlit as st
import pandas as pd

# If you already have OUTPUT_DIR in config.py, import it; else fallback
# try:
#     from config import OUTPUT_DIR as DEFAULT_OUTPUT_DIR
# except Exception:
DEFAULT_OUTPUT_DIR = str(Path("Data/Output").resolve())

st.set_page_config(page_title="Document Viewer — Transparency & Explainability", layout="wide")

# ------------- Helpers -------------
def list_docs_by_type(root: Path) -> dict[str, list[Path]]:
    """
    Assumes your folder structure:
      <root>/pos/*.pdf
      <root>/invoices/*.pdf
      <root>/grns/*.(csv|pdf)
    Returns dict with keys 'PO', 'INV', 'GRN'
    """
    root = Path(root)
    return {
        "PO" : sorted((root / "pos").rglob("*.pdf")),
        "INV": sorted((root / "invoices").rglob("*.pdf")),
        "GRN": sorted((root / "grns").rglob("*.csv")),
    }

def render_pdf_inline(pdf_path: Path, height: int = 800):
    """
    Renders a PDF (selectable text, scrollable) using a data URL in an iframe.
    """
    with open(pdf_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")
    html = f"""
        <iframe
            src="data:application/pdf;base64,{b64}#view=FitH"
            style="width:100%; height:{height}px; border:none;"
        ></iframe>
    """
    st.components.v1.html(html, height=height, scrolling=True)

def clean_table(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # 1) Drop 'exceptions' column if present
    for col in df.columns:
        if str(col).strip().lower() == "exceptions":
            df.drop(columns=[col], inplace=True)
            break

    # 2) Remove rows that are actually header echoes
    #    (row values look like the column names)
    norm_cols = [str(c).strip().lower() for c in df.columns]

    def is_header_echo(row) -> bool:
        vals = [str(v).strip().lower() for v in row.values.tolist()]
        # count how many cells exactly match a column name
        matches = sum(v in norm_cols for v in vals)
        # consider it a header-echo if >= 50% of columns match (min 3)
        return matches >= max(3, int(0.5 * len(norm_cols)))

    mask = df.apply(is_header_echo, axis=1)
    df = df.loc[~mask]

    # 3) Drop fully-empty rows and reindex
    df = df[~df.isna().all(axis=1)].reset_index(drop=True)
    return df

def load_if_exists(csv_path: Path) -> pd.DataFrame | None:
    try:
        if csv_path.exists():
            df = pd.read_csv(csv_path)
            return clean_table(df)   # <-- apply cleaner here
    except Exception as e:
        st.warning(f"Could not read {csv_path.name}: {type(e).__name__}: {e}")
    return None

# ------------- Page 2 UI -------------
st.title("Document Viewer — Transparency & Explainability")

# -------- Section 1: Select & render original document --------
st.subheader("Original Document")

# Let user pick the input root (main area; no sidebar)
data_root = st.text_input(
    "Input data root (same structure as Page 1)",
    value=str(Path("Data/incoming_sample").resolve()),
    help="Must contain subfolders: pos/, invoices/, grns/",
)

files_map = {"PO": [], "INV": [], "GRN": []}
if data_root:
    files_map = list_docs_by_type(Path(data_root))

col1, col2 = st.columns([1, 3], vertical_alignment="bottom")
with col1:
    doc_type = st.selectbox("Document type", options=["GRN", "PO", "INV"])
with col2:
    choices = files_map.get(doc_type, [])
    nice_names = [p.name for p in choices] if choices else []
    selected_name = st.selectbox(
        "File name",
        options=nice_names if nice_names else ["— no files found —"],
        index=0,
        disabled=not bool(nice_names),
    )

if choices and selected_name and selected_name != "— no files found —":
    chosen_path = next(p for p in choices if p.name == selected_name)
    st.caption(f"Path: {chosen_path}")
    if chosen_path.suffix.lower() == ".pdf":
        render_pdf_inline(chosen_path, height=820)
    else:
        # GRN (CSV) fallback: show as table since it's not a PDF
        st.info("Rendering CSV in place (original GRN).")
        try:
            df = pd.read_csv(chosen_path)
            st.dataframe(df, use_container_width=True, height=420)
        except Exception as e:
            st.error(f"Failed to render CSV: {type(e).__name__}: {e}")
else:
    st.info("Select a document type and file to render the original.")

st.divider()

# -------- Section 2: Output tables (5 dataframes) --------
st.subheader("Normalized Tables")

# Output directory (main area; no sidebar)
output_dir = st.text_input(
    "Processed output directory",
    value=str(Path(DEFAULT_OUTPUT_DIR).resolve()),
    help="Folder where Page 1 wrote the CSVs (po_agg_df, po_line_df, inv_agg_df, inv_line_df, grn_line_df).",
)

# Load tables if present
out = Path(output_dir)
po_agg = load_if_exists(out / "po_agg_df.csv")
po_line = load_if_exists(out / "po_line_df.csv")
inv_agg = load_if_exists(out / "inv_agg_df.csv")
inv_line = load_if_exists(out / "inv_line_df.csv")
grn_line = load_if_exists(out / "grn_line_df.csv")

po_agg = po_agg[po_agg['total_amount_usd'].notna()].reset_index(drop=True)
po_line = po_line[po_line['line_total_usd'].notna()].reset_index(drop=True)
inv_agg = inv_agg[inv_agg['total_amount_usd'].notna()].reset_index(drop=True)
inv_line = inv_line[inv_line['line_total_usd'].notna()].reset_index(drop=True)

tabs = st.tabs([
    "PO_Aggregate", "PO_LineItem", "INV_Aggregate", "INV_LineItem", "GRN_LineItem"
])

with tabs[0]:
    st.caption(
    "**Purpose:** Header-level PO data for financial matching.\n\n"
    "**Primary Key:** `po_number`\n\n"
    "**Key Columns:** `po_number` (PK), `doc_date`, `vendor_id`, `total_amount`, `total_amount_usd`, `exceptions`."
)

    if po_agg is not None and not po_agg.empty:
        st.dataframe(po_agg.drop_duplicates().reset_index(drop=True), use_container_width=True, height=360)
    else:
        st.info("po_agg_df.csv not found or empty.")

with tabs[1]:
    st.caption(
    "**Purpose:** Line-level PO data (\"what was ordered\") for quantity matching.\n\n"
    "**Primary Key:** (`po_number`, `line_number`)\n\n"
    "**Key Columns:** `po_number` (FK), `line_number`, `sku_id`, `description`, `quantity`, `unit_price`, `line_total`."
)

    if po_line is not None and not po_line.empty:
        st.dataframe(po_line.drop_duplicates().reset_index(drop=True), use_container_width=True, height=360)
    else:
        st.info("po_line_df.csv not found or empty.")

with tabs[2]:
    st.caption(
    "**Purpose:** Header-level Invoice data for financial matching against PO_Aggregate.\n\n"
    "**Primary Key:** `invoice_number`\n\n"
    "**Key Columns:** `invoice_number` (PK), `po_number` (FK), `doc_date`, `vendor_id`, `total_amount`, `total_amount_usd`, `exceptions`.\n\n"
    "_Note: `lines_json` is removed; lines are in their own table._"
)

    if inv_agg is not None and not inv_agg.empty:
        st.dataframe(inv_agg.drop_duplicates().reset_index(drop=True), use_container_width=True, height=360)
    else:
        st.info("inv_agg_df.csv not found or empty.")

with tabs[3]:
    st.caption(
    "**Purpose:** Line-level Invoice data (\"what was billed\") for detailed reconciliation against PO_LineItem and GRN_LineItem.\n\n"
    "**Primary Key:** (`invoice_number`, `line_number`)\n\n"
    "**Key Columns:** `invoice_number` (FK), `line_number`, `sku_id`, `description`, `quantity`, `unit_price`, `line_total`."
)

    if inv_line is not None and not inv_line.empty:
        st.dataframe(inv_line.drop_duplicates().reset_index(drop=True), use_container_width=True, height=360)
    else:
        st.info("inv_line_df.csv not found or empty.")

with tabs[4]:
    st.caption(
    "**Purpose:** Line-level GRN data (\"what was received\") for quantity matching against PO_LineItem.\n\n"
    "**Primary Key:** (`grn_number`, `line_number`)\n\n"
    "**Key Columns:** `grn_number` (PK/FK), `po_number` (FK), `doc_date`, `line_number`, `sku_id`, `description`, `quantity`."
)

    if grn_line is not None and not grn_line.empty:
        st.dataframe(grn_line.drop_duplicates().reset_index(drop=True), use_container_width=True, height=360)
    else:
        st.info("grn_line_df.csv not found or empty.")

st.warning("Please click on Reconciliation in the sidebar to proceed with automated 3-way reconciliation.")

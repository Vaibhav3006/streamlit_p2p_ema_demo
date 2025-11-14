# streamlit_adapter.py
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
import hashlib
import math
import pandas as pd

from config import WORKERS, BATCH_SIZE, AWS_REGION #INPUT_DIRS, OUTPUT_DIR, 
from extraction import process_one
from extraction_llm import extract_receipt_details
from normalization import normalize_document_advanced
from final_tables import DocumentProcessor

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

FX_RATE_FILE = Path(__file__).resolve() / "Data" / "fx_rates_sample.csv"
# FX_RATE_FILE = str(Path("Data/fx_rates_sample.csv").resolve())
#INPUT_DIRS = str(Path("Data/incoming_sample").resolve())
#OUTPUT_DIR = str(Path("Data/Output").resolve())

def file_checksum(fp: Path, block_size: int = 65536) -> str:
    h = hashlib.md5()
    with open(fp, "rb") as f:
        for chunk in iter(lambda: f.read(block_size), b""):
            h.update(chunk)
    return h.hexdigest()

def iter_files(paths, exts=(".pdf", ".csv")):
    for p in paths:
        base = Path(p)
        for f in base.rglob("*"):
            if f.suffix.lower() in exts:
                yield f

def is_pdf(fp: Path) -> bool:
    return fp.suffix.lower() == ".pdf"

def is_csv(fp: Path) -> bool:
    return fp.suffix.lower() == ".csv"

def process_file_for_meta(file_path: Path):
    """
    Runs your per-file pipeline and returns:
      final tables + a small meta record for KPIs/logs.
    """
    start_ts = datetime.utcnow()
    obj = process_one(str(file_path), region_name=AWS_REGION)

    payload = str(obj)  # safe for both PDF/CSV in your current setup
    response_text = extract_receipt_details(payload)

    normalized = normalize_document_advanced(response_text, FX_RATE_FILE)

    proc = DocumentProcessor()
    proc.add_document(normalized)
    proc.finalize_tables()
    po_agg_df, po_line_df, inv_agg_df, inv_line_df, grn_line_df = proc.get_tables()

    # --- derive lightweight meta for KPIs
    # try to infer doc_type from any table populated
    if not inv_agg_df.empty or not inv_line_df.empty:
        doc_type = "INV"
    elif not po_agg_df.empty or not po_line_df.empty:
        doc_type = "PO"
    elif not grn_line_df.empty:
        doc_type = "GRN"
    else:
        doc_type = "UNKNOWN"

    # infer ocr confidence if available in normalized payload or tables
    # (fall back to None if absent)
    ocr_conf = None
    for df in [po_agg_df, inv_agg_df, grn_line_df]:
        if not df.empty:
            for col in df.columns:
                if col.lower() in {"ocr_confidence", "avg_ocr_confidence"}:
                    try:
                        ocr_conf = float(df[col].astype(float).mean())
                    except Exception:
                        pass
            break

    # infer "auto-match" from any known column if present
    # expect a boolean/flag/enum like: match_status == "FULL"
    auto_match = None
    for df in [inv_agg_df, inv_line_df]:
        if not df.empty:
            cols = {c.lower() for c in df.columns}
            if "match_status" in cols:
                ms = df[[c for c in df.columns if c.lower() == "match_status"][0]].astype(str)
                auto_match = (ms.str.upper() == "FULL").mean()
                break
            if "auto_match" in cols:
                col = [c for c in df.columns if c.lower() == "auto_match"][0]
                auto_match = df[col].astype(bool).mean()
                break

    # simple exception counts if table has severity columns
    exceptions_critical = 0
    exceptions_warning = 0
    for df in [inv_agg_df, inv_line_df, po_agg_df, po_line_df, grn_line_df]:
        if not df.empty:
            cols = {c.lower() for c in df.columns}
            if "exception_severity" in cols:
                col = [c for c in df.columns if c.lower() == "exception_severity"][0]
                sev = df[col].astype(str).str.lower()
                exceptions_critical += (sev == "critical").sum()
                exceptions_warning += (sev == "warning").sum()

    duration_s = (datetime.utcnow() - start_ts).total_seconds()

    meta = {
        "file": str(file_path),
        "doc_type": doc_type,
        "start_utc": start_ts.isoformat(timespec="seconds") + "Z",
        "duration_s": duration_s,
        "ocr_confidence": ocr_conf,         # float | None
        "auto_match_rate": auto_match,      # float in [0,1] | None
        "exceptions_critical": exceptions_critical,
        "exceptions_warning": exceptions_warning,
    }

    return (po_agg_df, po_line_df, inv_agg_df, inv_line_df, grn_line_df), meta

def run_parallel_streaming(
    input_dirs=None,
    output_dir=None,
    workers=None,
    batch_size=None,
):
    """
    Generator for Streamlit: yields per-batch status dicts.
    """
    input_dirs = input_dirs or INPUT_DIRS
    output_dir = output_dir or OUTPUT_DIR
    workers = workers or WORKERS
    batch_size = batch_size or BATCH_SIZE
    
    # if files is None:
    #     files = list(iter_files(input_dirs, exts=(".pdf", ".csv")))
    # else:
    #     # normalize to Paths
    #     files = [Path(f) for f in files]

    files = list(iter_files(input_dirs, exts=(".pdf", ".csv")))
    

    total = len(files)
    logger.info(f"total files: {total}")
    if total == 0:
        yield {"event": "init", "total_files": 0}
        return

    num_batches = math.ceil(total / batch_size)
    header_written = {
        "po_agg_df": False, "po_line_df": False,
        "inv_agg_df": False, "inv_line_df": False,
        "grn_line_df": False
    }

    # init
    yield {"event": "init", "total_files": total, "num_batches": num_batches}

    processed = 0
    for bi in range(num_batches):
        start = bi * batch_size
        end = min((bi + 1) * batch_size, total)
        batch_files = files[start:end]
        batch_id = f"Batch_{bi+1:02d}"

        batch_tables = {
            "po_agg_df": pd.DataFrame(),
            "po_line_df": pd.DataFrame(),
            "inv_agg_df": pd.DataFrame(),
            "inv_line_df": pd.DataFrame(),
            "grn_line_df": pd.DataFrame(),
        }
        metas = []
        errors = []
        batch_start = datetime.utcnow()

        with ThreadPoolExecutor(max_workers=workers) as ex:
            fut_map = {ex.submit(process_file_for_meta, fp): fp for fp in batch_files}
            for fut in as_completed(fut_map):
                fp = fut_map[fut]
                try:
                    (po_agg, po_line, inv_agg, inv_line, grn_line), meta = fut.result()
                    if not po_agg.empty:
                        batch_tables["po_agg_df"] = pd.concat([batch_tables["po_agg_df"], po_agg], ignore_index=True)
                    if not po_line.empty:
                        batch_tables["po_line_df"] = pd.concat([batch_tables["po_line_df"], po_line], ignore_index=True)
                    if not inv_agg.empty:
                        batch_tables["inv_agg_df"] = pd.concat([batch_tables["inv_agg_df"], inv_agg], ignore_index=True)
                    if not inv_line.empty:
                        batch_tables["inv_line_df"] = pd.concat([batch_tables["inv_line_df"], inv_line], ignore_index=True)
                    if not grn_line.empty:
                        batch_tables["grn_line_df"] = pd.concat([batch_tables["grn_line_df"], grn_line], ignore_index=True)

                    metas.append(meta)
                except Exception as e:
                    errors.append({"file": str(fp), "error": f"{type(e).__name__}: {e}"})

        # write-out
        out_dir = Path(output_dir); out_dir.mkdir(parents=True, exist_ok=True)
        for name, df in batch_tables.items():
            if df is None or df.empty:
                continue
            out_path = out_dir / f"{name}.csv"
            write_header = not header_written.get(name, out_path.exists())
            df.to_csv(out_path, mode="a", index=False, header=write_header)
            header_written[name] = True

        processed += len(batch_files)
        dur = (datetime.utcnow() - batch_start).total_seconds()

        # summarize batch
        doc_counts = pd.Series([m["doc_type"] for m in metas]).value_counts().to_dict() if metas else {}
        avg_ocr = pd.Series([m["ocr_confidence"] for m in metas if m["ocr_confidence"] is not None]).mean() if metas else None
        avg_auto = pd.Series([m["auto_match_rate"] for m in metas if m["auto_match_rate"] is not None]).mean() if metas else None
        exc_c = sum(m["exceptions_critical"] for m in metas)
        exc_w = sum(m["exceptions_warning"] for m in metas)

        yield {
            "event": "batch_done",
            "batch_id": batch_id,
            "batch_size": len(batch_files),
            "duration_s": dur,
            "doc_counts": doc_counts,  # e.g., {"PO": 10, "INV": 15, "GRN": 12}
            "avg_ocr_confidence": "NA", #float(avg_ocr) if avg_ocr == avg_ocr else None,
            "avg_auto_match_rate": "NA", #float(avg_auto) if avg_auto == avg_auto else None,
            "exceptions_critical": int(exc_c),
            "exceptions_warning": int(exc_w),
            "processed": processed,
            "errors": errors,
        }
